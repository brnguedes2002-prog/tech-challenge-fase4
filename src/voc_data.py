"""Read-only data loading. Remote gold is preserved in a local auditable snapshot."""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

GOLD_COLUMNS = ["review_id", "review_full_text", "review_score", "order_status",
                "order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date", "product_category_name_english"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value):
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def load_local_gold(directory: Path, *, require_message=False):
    """Parse original multiline CSV correctly; joins never multiply reviews."""
    def read(name):
        return pd.read_csv(directory / name, dtype=str, keep_default_na=False)
    reviews = read("olist_order_reviews_dataset.csv")
    raw_count = len(reviews)
    reviews["review_full_text"] = (reviews["review_comment_title"].map(clean_text) + " " +
                                   reviews["review_comment_message"].map(clean_text)).str.strip()
    has_text = reviews.review_comment_message.map(clean_text).ne("") if require_message else reviews.review_full_text.ne("")
    reviews = reviews[has_text & reviews.review_id.ne("")].copy()
    # Stable policy documented in provenance, independent of arbitrary join order.
    reviews = reviews.drop_duplicates("review_id", keep="first")
    orders = read("olist_orders_dataset.csv").drop_duplicates("order_id")
    cols = ["order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date",
            "order_estimated_delivery_date"]
    gold = reviews.merge(orders[cols], on="order_id", how="left", validate="many_to_one")
    items = read("olist_order_items_dataset.csv")
    items["_item_order"] = pd.to_numeric(items["order_item_id"], errors="coerce")
    items = items.sort_values(["order_id", "_item_order", "product_id"], kind="stable").drop_duplicates("order_id")
    products = read("olist_products_dataset.csv").drop_duplicates("product_id")
    trans = read("product_category_name_translation.csv").drop_duplicates("product_category_name")
    categories = items[["order_id", "product_id"]].merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left", validate="many_to_one"
    ).merge(trans, on="product_category_name", how="left", validate="many_to_one")
    gold = gold.merge(categories[["order_id", "product_category_name_english"]], on="order_id", how="left", validate="many_to_one")
    gold["review_score"] = pd.to_numeric(gold["review_score"], errors="coerce")
    gold = gold[GOLD_COLUMNS].reset_index(drop=True)
    digest = hashlib.sha256(b"message-required\0" if require_message else b"")
    for name in sorted(["olist_order_reviews_dataset.csv", "olist_orders_dataset.csv", "olist_order_items_dataset.csv",
                        "olist_products_dataset.csv", "product_category_name_translation.csv"]):
        digest.update(name.encode())
        with (directory / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return gold, {"source": "CSVs originais Olist locais", "dataset_version": digest.hexdigest(),
                  "loaded_at": utc_now(), "raw_reviews": raw_count, "reviews": len(gold),
                  "policy": ("CSV multilinha; mensagem obrigatória; " if require_message else "CSV multilinha; título ou mensagem; ")
                  + "primeiro review_id; categoria do primeiro item por pedido"}


def quote_table(table):
    if not re.fullmatch(r"[A-Za-z_][\w]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*", table, flags=re.ASCII):
        raise ValueError("Nome de tabela deve ser catalogo.schema.tabela, sem SQL adicional.")
    return ".".join(f"`{part}`" for part in table.split("."))


def execute_read(client, warehouse_id, sql, timeout=180):
    """SDK Statement Execution, JSON INLINE; only caller-owned SELECT statements."""
    response = client.statement_execution.execute_statement(
        statement=sql, warehouse_id=warehouse_id, wait_timeout="10s").as_dict()
    deadline = time.monotonic() + timeout
    statement_id = response.get("statement_id")
    while response.get("status", {}).get("state") in ("PENDING", "RUNNING"):
        if time.monotonic() >= deadline:
            raise TimeoutError("A leitura SQL excedeu o tempo disponível.")
        time.sleep(1)
        response = client.statement_execution.get_statement(statement_id).as_dict()
    if response.get("status", {}).get("state") != "SUCCEEDED":
        # SQL details can include workspace identifiers or sensitive service messages.
        raise RuntimeError("Databricks não concluiu a leitura SQL. Verifique warehouse e SELECT na tabela.")
    manifest = response.get("manifest") or {}
    if manifest.get("truncated"):
        raise RuntimeError("Resultado SQL truncado; snapshot incompleto foi rejeitado.")
    columns = [c["name"] for c in manifest.get("schema", {}).get("columns", [])]
    result = response.get("result") or {}
    rows = list(result.get("data_array") or [])
    seen = set()
    while result.get("next_chunk_index") is not None:
        index = result["next_chunk_index"]
        if index in seen:
            raise RuntimeError("Paginação SQL repetida; leitura interrompida.")
        seen.add(index)
        result = client.statement_execution.get_statement_result_chunk_n(statement_id, index).as_dict()
        rows.extend(result.get("data_array") or [])
    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs["source_schema"] = manifest.get("schema", {}).get("columns", [])
    return frame


def load_remote_gold(client, warehouse_id, table, snapshot_dir: Path, progress=lambda _: None):
    quoted = quote_table(table)
    # Count and uniqueness guard are read-only and detect keyset pagination hazards.
    counts = execute_read(client, warehouse_id,
                          f"SELECT count(*) AS total, count(DISTINCT review_id) AS ids FROM {quoted}")
    total, unique = int(counts.iloc[0]["total"]), int(counts.iloc[0]["ids"])
    if total != unique:
        raise ValueError("Gold contém review_id nulo ou duplicado. Paginação segura exige IDs únicos.")
    if not total:
        raise ValueError("A tabela gold está vazia.")
    frames, cursor, received = [], None, 0
    while received < total:
        # Hex encoding keeps even malformed IDs out of SQL syntax (quotes/backslashes).
        where = "" if cursor is None else " WHERE review_id > CAST(unhex('" + cursor.encode("utf-8").hex() + "') AS STRING)"
        frame = execute_read(client, warehouse_id, f"SELECT * FROM {quoted}{where} ORDER BY review_id LIMIT 1000")
        if frame.empty:
            raise RuntimeError("Gold mudou durante a leitura; snapshot incompleto rejeitado.")
        frames.append(frame)
        received += len(frame)
        cursor = str(frame.iloc[-1]["review_id"])
        progress(f"Copiando gold para o computador: {received:,} de {total:,} avaliações.")
    gold = pd.concat(frames, ignore_index=True)
    final_count = int(execute_read(client, warehouse_id, f"SELECT count(*) AS total FROM {quoted}").iloc[0]["total"])
    if len(gold) != total or final_count != total or gold.review_id.duplicated().any():
        raise RuntimeError("A contagem da gold mudou durante a cópia. Tente novamente.")
    missing = set(GOLD_COLUMNS) - set(gold.columns)
    if missing:
        raise ValueError("Gold sem colunas exigidas pelo pipeline: " + ", ".join(sorted(missing)))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    payload = gold.to_json(orient="records", lines=True, force_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path = snapshot_dir / "gold.jsonl"
    temp = snapshot_dir / "gold.jsonl.tmp"
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)
    provenance = {"source": table, "host": getattr(client.config, "host", None),
                  "loaded_at": utc_now(), "dataset_version": digest, "reviews": len(gold),
                  "columns": list(gold.columns), "snapshot_file": "gold.jsonl",
                  "source_schema": frames[0].attrs.get("source_schema", []),
                  "consistency": "Contagens e unicidade verificadas; não é snapshot transacional entre páginas.",
                  "policy": "Cópia da gold atual sem correção de ingestão; temas calculados somente localmente."}
    (snapshot_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return gold, provenance


def prepare_gold(gold):
    """Coerce SQL JSON scalar types for the existing pipeline; preserve every row."""
    frame = gold.copy()
    if frame.review_id.isna().any() or frame.review_id.duplicated().any():
        raise ValueError("O pipeline exige review_id presente e único.")
    frame["review_score"] = pd.to_numeric(frame["review_score"], errors="coerce")
    invalid_scores = int((~frame.review_score.between(1, 5) | frame.review_score.isna()).sum())
    # Keep invalid scores as unknown, never map them to a satisfaction class.
    frame.loc[~frame.review_score.between(1, 5), "review_score"] = float("nan")
    if frame.review_full_text.isna().any():
        raise ValueError("Gold contém texto nulo; não substituir conteúdo silenciosamente.")
    frame["review_full_text"] = frame.review_full_text.astype(str)
    return frame, {"invalid_or_missing_scores": invalid_scores, "reviews": len(frame)}
