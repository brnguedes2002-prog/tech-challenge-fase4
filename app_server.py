"""Local-only web application. Databricks access is read/query only."""
from __future__ import annotations
import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from voc_backend import Backend, BackendError, EXAMPLES
from app_config import load_env


def handler_for(backend, webroot):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VoCLocal/1.0"

        def _host_ok(self):
            return self.headers.get("Host", "") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def _headers(self, status, content_type, size):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()

        def _json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _error(self, code, message, status):
            self._json({"error": {"code": code, "message": message}}, status)

        def do_GET(self):
            if not self._host_ok():
                return self._error("INVALID_HOST", "Acesso permitido somente pelo endereço local.", 403)
            path = urlsplit(self.path).path
            if path == "/api/status":
                return self._json(backend.status())
            if path == "/api/examples":
                return self._json({"examples": EXAMPLES})
            if path.startswith("/api/"):
                return self._error("NOT_FOUND", "Recurso não encontrado.", 404)
            target = (webroot / (unquote(path).lstrip("/") or "index.html")).resolve()
            if not target.is_relative_to(webroot.resolve()) or not target.is_file():
                return self._error("NOT_FOUND", "Arquivo não encontrado.", 404)
            body = target.read_bytes()
            mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
                mime += "; charset=utf-8"
            self._headers(200, mime, len(body))
            self.wfile.write(body)

        def do_POST(self):
            if not self._host_ok():
                return self._error("INVALID_HOST", "Host não permitido.", 403)
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                return self._error("INVALID_ORIGIN", "Origem não permitida.", 403)
            if urlsplit(self.path).path != "/api/query":
                return self._error("NOT_FOUND", "Recurso não encontrado.", 404)
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                return self._error("INVALID_CONTENT_TYPE", "Envie JSON.", 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._error("INVALID_BODY", "Tamanho inválido.", 400)
            if size <= 0 or size > 10240:
                return self._error("INVALID_BODY", "Limite de 10 KB por consulta.", 413)
            try:
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError()
            except (ValueError, UnicodeDecodeError):
                return self._error("INVALID_JSON", "Corpo JSON inválido.", 400)
            try:
                return self._json(backend.query(payload.get("question")))
            except BackendError as exc:
                return self._error(exc.code, exc.message, exc.status)

        def log_message(self, format, *args):
            # Never log request bodies, credentials, query string or model outputs.
            pass
    return Handler


def main():
    load_env(ROOT / '.env')
    parser = argparse.ArgumentParser(description="Voice of Customer: aplicação somente local")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    backend = Backend(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(backend, ROOT / "web"))
    server.daemon_threads = True
    backend.start()
    print(f"Aplicação local: http://127.0.0.1:{args.port} | modo: {backend.mode}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Encerrando servidor local.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
