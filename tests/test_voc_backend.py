import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from voc_backend import Backend, BackendError
from app_server import handler_for


class BackendTests(unittest.TestCase):
    def ready_backend(self, mode="local"):
        backend = Backend(ROOT, env={"VOC_MODE": mode})
        backend.pipeline = Mock()
        backend.pipeline.answer.return_value.to_dict.return_value = {"resposta": "Trecho [R1]", "avisos": [], "abstencao": False}
        backend.provenance = {"dataset_version": "version"}
        backend._status.update(ready=True, configuration={"min_score": .6}, limitations=["limite"])
        return backend

    def test_local_mode_cannot_claim_llm(self):
        result = self.ready_backend().query("entrega")
        self.assertEqual(result["meta"]["mode"], "local")
        self.assertIn("não é geração por LLM", result["response"]["avisos"][0])

    def test_live_failure_never_falls_back(self):
        backend = self.ready_backend("databricks")
        backend.pipeline.answer.side_effect = RuntimeError("secret-token")
        with self.assertRaises(BackendError) as error:
            backend.query("entrega")
        self.assertEqual(error.exception.code, "QUERY_FAILED")
        self.assertNotIn("secret-token", error.exception.message)
        self.assertEqual(backend.mode, "databricks")

    def test_busy_query_does_not_call_pipeline(self):
        backend = self.ready_backend()
        backend._query_lock.acquire()
        try:
            with self.assertRaises(BackendError) as error:
                backend.query("entrega")
            self.assertEqual(error.exception.status, 409)
            backend.pipeline.answer.assert_not_called()
        finally:
            backend._query_lock.release()

    def test_question_validation(self):
        backend = self.ready_backend()
        for q in (None, [], " ", "a" * 2001):
            with self.assertRaises(BackendError):
                backend.query(q)
        backend.pipeline.answer.assert_not_called()


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "index.html").write_text("<h1>VoC</h1>", encoding="utf-8")
        self.backend = Mock()
        self.backend.status.return_value = {"ready": True}
        self.backend.query.return_value = {"response": {"resposta": "ok"}, "meta": {"mode": "local"}}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.backend, root))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def request(self, path, method="GET", body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        conn.request(method, path, body=body, headers=headers or {})
        result = conn.getresponse()
        response = result.status, result.read()
        conn.close()
        return response

    def test_api_and_query(self):
        self.assertEqual(self.request("/api/status")[0], 200)
        status, body = self.request("/api/query", "POST", json.dumps({"question": "entrega"}), {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.backend.query.assert_called_once_with("entrega")

    def test_cross_origin_and_host_blocked(self):
        self.assertEqual(self.request("/api/status", headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.request("/api/query", "POST", "{}", {"Content-Type": "application/json", "Origin": "https://evil.example"})[0], 403)
        self.backend.query.assert_not_called()

    def test_static_traversal_not_served(self):
        self.assertEqual(self.request("/%2e%2e/app_server.py")[0], 404)

    def test_large_and_invalid_body(self):
        self.assertEqual(self.request("/api/query", "POST", "x" * 10241, {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("/api/query", "POST", "[1]", {"Content-Type": "application/json"})[0], 400)
        self.backend.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
