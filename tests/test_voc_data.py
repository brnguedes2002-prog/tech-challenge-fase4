import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import voc_data as vd


class DataTests(unittest.TestCase):
    def write_csv(self, folder, filename, rows):
        with (folder / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_multiline_and_multi_item_join_do_not_multiply_reviews(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_csv(root, "olist_order_reviews_dataset.csv", [
                dict(review_id="r1", order_id="o1", review_score="1", review_comment_title="Entrega", review_comment_message='não chegou\nsegunda linha, com "aspas"'),
                dict(review_id="r1", order_id="o1", review_score="1", review_comment_title="Duplicata", review_comment_message="ignorada"),
                dict(review_id="r2", order_id="o2", review_score="5", review_comment_title="", review_comment_message=""),
            ])
            self.write_csv(root, "olist_orders_dataset.csv", [dict(order_id="o1", order_status="delivered", order_purchase_timestamp="2018-01-01", order_delivered_customer_date="2018-01-05", order_estimated_delivery_date="2018-01-03")])
            self.write_csv(root, "olist_order_items_dataset.csv", [dict(order_id="o1", order_item_id="2", product_id="p2"), dict(order_id="o1", order_item_id="1", product_id="p1")])
            self.write_csv(root, "olist_products_dataset.csv", [dict(product_id="p1", product_category_name="cama"), dict(product_id="p2", product_category_name="livros")])
            self.write_csv(root, "product_category_name_translation.csv", [dict(product_category_name="cama", product_category_name_english="bed_bath_table"), dict(product_category_name="livros", product_category_name_english="books")])
            gold, meta = vd.load_local_gold(root)
            self.assertEqual(len(gold), 1)
            self.assertIn('segunda linha, com "aspas"', gold.iloc[0].review_full_text)
            self.assertEqual(gold.iloc[0].product_category_name_english, "bed_bath_table")
            self.assertEqual(meta["raw_reviews"], 3)
            self.assertEqual(len(meta["dataset_version"]), 64)

    def test_prepare_preserves_bad_scores_as_unknown(self):
        frame = pd.DataFrame({"review_id": ["a", "b", "c"], "review_score": ["1", None, "999"], "review_full_text": ["x", "y", "z"]})
        gold, quality = vd.prepare_gold(frame)
        self.assertEqual(len(gold), 3)
        self.assertEqual(quality["invalid_or_missing_scores"], 2)
        self.assertTrue(pd.isna(gold.iloc[2].review_score))
        self.assertEqual(frame.iloc[2].review_score, "999")

    def test_reject_sql_identifier_injection(self):
        with self.assertRaises(ValueError):
            vd.quote_table("workspace.raw.gold;DROP TABLE x")
        self.assertEqual(vd.quote_table("workspace.raw.gold"), "`workspace`.`raw`.`gold`")

    def test_statement_fetches_all_chunks(self):
        client = Mock()
        client.statement_execution.execute_statement.return_value.as_dict.return_value = {
            "statement_id": "s1", "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": "review_id"}]}},
            "result": {"data_array": [["a"]], "next_chunk_index": 1}}
        client.statement_execution.get_statement_result_chunk_n.return_value.as_dict.return_value = {"data_array": [["b"]]}
        frame = vd.execute_read(client, "warehouse", "SELECT review_id FROM gold")
        self.assertEqual(frame.review_id.tolist(), ["a", "b"])
        client.statement_execution.get_statement_result_chunk_n.assert_called_once_with("s1", 1)

    def test_truncated_statement_rejected(self):
        client = Mock()
        client.statement_execution.execute_statement.return_value.as_dict.return_value = {
            "status": {"state": "SUCCEEDED"}, "manifest": {"truncated": True}}
        with self.assertRaises(RuntimeError):
            vd.execute_read(client, "warehouse", "SELECT * FROM gold")

    def test_remote_snapshot_full_and_no_data_writes(self):
        rows = [{c: None for c in vd.GOLD_COLUMNS} for _ in range(2)]
        for row, rid in zip(rows, ["a'\\", "z"]):
            row.update(review_id=rid, review_full_text="avaliação", review_score="1")
        responses = [pd.DataFrame([dict(total="2", ids="2")]), pd.DataFrame(rows[:1]), pd.DataFrame(rows[1:]), pd.DataFrame([dict(total="2")])]
        client = SimpleNamespace(config=SimpleNamespace(host="https://workspace.example"))
        with tempfile.TemporaryDirectory() as temp, patch.object(vd, "execute_read", side_effect=responses) as call:
            gold, meta = vd.load_remote_gold(client, "wh", "workspace.raw.gold", Path(temp))
            self.assertEqual(len(gold), 2)
            self.assertTrue((Path(temp) / "gold.jsonl").is_file())
            self.assertEqual(json.loads((Path(temp) / "provenance.json").read_text(encoding="utf-8"))["reviews"], 2)
            sqls = [c.args[2] for c in call.call_args_list]
            self.assertTrue(all(s.startswith("SELECT ") for s in sqls))
            self.assertIn("unhex('61275c')", sqls[2])
            self.assertEqual(meta["reviews"], 2)


if __name__ == "__main__":
    unittest.main()
