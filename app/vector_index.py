"""Full-corpus Airflow issue embedding index stored in existing Databricks Delta.

Usage (inside the deployed container):
    python -m app.vector_index
    python -m app.vector_index --max-issues 100

A resumable, explicitly invoked administrative batch job: never exposed as an
HTTP endpoint. Only changed/new issues are embedded on subsequent runs.
"""
import argparse
import hashlib
import json
import logging
import math
import time
import urllib.request
import urllib.error

from databricks import sql
from openai import OpenAI

from .config import settings

logger = logging.getLogger(__name__)
TABLE = f"{settings.databricks_catalog}.{settings.databricks_schema}.github_issue_embeddings"
ISSUES = f"{settings.databricks_catalog}.{settings.databricks_schema}.github_issues"
EMBEDDING_MODEL = settings.azure_openai_embedding_deployment
PAGE_SIZE = 500
BATCH_SIZE = 32


def _document(issue: dict) -> str:
    return ((issue.get("title") or "")[:300] + "\n" + (issue.get("body") or "")[:1700]).strip()


def _normalize(vector: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vector))
    if not n:
        raise ValueError("Received zero-length embedding vector")
    return [round(float(x / n), 8) for x in vector]


def _write_statement(query, parameters=()):
    host = settings.databricks_server_hostname.rstrip("/")
    if not host.startswith("https://"):
        host = "https://" + host
    url = host + "/api/2.0/sql/statements"
    warehouse = settings.databricks_http_path.rstrip("/").split("/")[-1]
    # Convert positional markers to named Statement Execution API parameters.
    for i in range(len(parameters)):
        query = query.replace("?", f":p{i}", 1)
    payload = {"warehouse_id": warehouse, "statement": query,
               "wait_timeout": "30s", "on_wait_timeout": "CONTINUE",
               "parameters": [{"name": f"p{i}", "value": str(v),
                               "type": "BIGINT" if isinstance(v, int) else "STRING"}
                              for i, v in enumerate(parameters)]}
    headers = {"Authorization": "Bearer " + settings.databricks_token,
               "Content-Type": "application/json"}
    def request(address, body=None):
        req = urllib.request.Request(address, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)
    response = request(url, json.dumps(payload).encode())
    deadline = time.monotonic() + 600
    while response["status"]["state"] in ("PENDING", "RUNNING"):
        if time.monotonic() >= deadline:
            raise TimeoutError("Databricks statement still pending; check query history before retry")
        time.sleep(2)
        response = request(url + "/" + response["statement_id"])
    if response["status"]["state"] != "SUCCEEDED":
        error = response["status"].get("error", {})
        raise RuntimeError("Databricks write failed: " + str(error.get("error_code", response["status"]["state"])))


def _index_batch(cursor, client, issues: list[dict]) -> None:
    documents = [_document(issue) or "(empty incident)" for issue in issues]
    result = client.embeddings.create(model=EMBEDDING_MODEL, input=documents)
    vectors = [item.embedding for item in sorted(result.data, key=lambda item: item.index)]
    if len(vectors) != len(issues):
        raise RuntimeError("Embedding service returned an unexpected batch size")

    items = []
    for issue, vector in zip(issues, vectors):
        items.extend([
            int(issue["issue_id"]),
            issue["_hash"],
            json.dumps(_normalize(vector), separators=(",", ":")),
        ])
    markers = ", ".join(["(?, ?, ?)"] * len(issues))
    query = f"""
    MERGE INTO {TABLE} target
    USING (
        SELECT CAST(issue_id AS BIGINT) issue_id, content_hash,
               from_json(embedding_json, 'ARRAY<FLOAT>') embedding
        FROM VALUES {markers} AS v(issue_id, content_hash, embedding_json)
    ) incoming
    ON target.issue_id = incoming.issue_id
    WHEN MATCHED AND target.content_hash <> incoming.content_hash THEN
      UPDATE SET content_hash = incoming.content_hash,
                 embedding = incoming.embedding, indexed_at = current_timestamp()
    WHEN NOT MATCHED THEN
      INSERT (issue_id, content_hash, embedding, indexed_at)
      VALUES (incoming.issue_id, incoming.content_hash,
              incoming.embedding, current_timestamp())
    """
    _write_statement(query, items)


def run(batch_size: int, max_issues: int) -> None:
    client = OpenAI(
        base_url=settings.azure_openai_endpoint.rstrip("/") + "/",
        api_key=settings.azure_openai_api_key,
        timeout=120.0,
        max_retries=3,
    )
    connection = sql.connect(
        server_hostname=settings.databricks_server_hostname,
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_token,
    )
    started = time.monotonic()
    seen = changed = indexed = 0
    pending = []
    try:
        with connection.cursor() as cur:
            _write_statement(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    issue_id BIGINT, content_hash STRING,
                    embedding ARRAY<FLOAT>, indexed_at TIMESTAMP
                ) USING DELTA
            """)
            cur.execute(f"SELECT issue_id, content_hash FROM {TABLE}")
            existing = {int(issue_id): digest for issue_id, digest in cur.fetchall()}
            print(f"OpsForge index: {len(existing):,} existing embeddings", flush=True)
            last_id = -1
            while True:
                cur.execute(f"""
                    SELECT issue_id, title, body FROM {ISSUES}
                    WHERE issue_id > ? ORDER BY issue_id ASC LIMIT {PAGE_SIZE}
                """, parameters=[last_id])
                rows = cur.fetchall()
                if not rows:
                    break
                for issue_id, title, body in rows:
                    last_id = int(issue_id)
                    seen += 1
                    issue = {"issue_id": issue_id, "title": title, "body": body}
                    issue["_hash"] = hashlib.sha256(
                        (_document(issue) + "\n" + EMBEDDING_MODEL).encode("utf-8")
                    ).hexdigest()
                    if existing.get(last_id) != issue["_hash"]:
                        changed += 1
                        pending.append(issue)
                        if len(pending) >= batch_size:
                            _index_batch(cur, client, pending)
                            indexed += len(pending)
                            pending.clear()
                            print(f"Processed {seen:,} issues; embedded {indexed:,}; "
                                  f"elapsed {time.monotonic()-started:.0f}s", flush=True)
                    if max_issues and seen >= max_issues:
                        break
                if max_issues and seen >= max_issues:
                    break
            if pending:
                _index_batch(cur, client, pending)
                indexed += len(pending)
                print(f"Processed {seen:,} issues; embedded {indexed:,}; "
                      f"elapsed {time.monotonic()-started:.0f}s", flush=True)
            print(f"Indexing completed: {seen:,} scanned, {changed:,} changed, "
                  f"{indexed:,} embedded, {time.monotonic()-started:.0f}s elapsed",
                  flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resumable full-corpus embedding job")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-issues", type=int, default=0,
                        help="0 = entire corpus; nonzero = first N issues")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64 or args.max_issues < 0:
        parser.error("batch-size must be 1–64 and max-issues nonnegative")
    logging.basicConfig(level=logging.INFO)
    run(args.batch_size, args.max_issues)
