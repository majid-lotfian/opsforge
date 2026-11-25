"""Read-only incident access and full-corpus semantic lookup in Delta."""
from databricks import sql
from .config import settings

def _query(query: str, parameters: list | None = None) -> list[dict]:
    with sql.connect(
        server_hostname=settings.databricks_server_hostname,
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_token,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters=parameters or [])
            columns = [c[0] for c in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

def search_issues(term: str, limit: int = 8) -> list[dict]:
    # Parameterized, bounded lexical retrieval; semantic indexing is a later extension.
    catalog = settings.databricks_catalog
    schema = settings.databricks_schema
    query = f"""
        SELECT issue_id, issue_number, title, body, state, url, updated_at
        FROM `{catalog}`.`{schema}`.`github_issues`
        WHERE lower(coalesce(title, '')) LIKE lower(?)
           OR lower(coalesce(body, '')) LIKE lower(?)
        ORDER BY updated_at DESC
        LIMIT {min(max(limit, 1), 20)}
    """
    pattern = "%" + term.replace("%", "\\%").replace("_", "\\_") + "%"
    return _query(query, [pattern, pattern])

def issue_comments(issue_id: int, limit: int = 30) -> list[dict]:
    catalog = settings.databricks_catalog
    schema = settings.databricks_schema
    return _query(f"""
        SELECT body, author_login, created_at, url
        FROM `{catalog}`.`{schema}`.`github_issue_comments`
        WHERE issue_id = ?
        ORDER BY created_at ASC
        LIMIT {min(max(limit, 1), 100)}
    """, [issue_id])


def _index_table() -> str:
    return f"{settings.databricks_catalog}.{settings.databricks_schema}.github_issue_embeddings"

def index_is_complete() -> bool:
    """Allow minor ingestion drift after the bulk index is available."""
    row = _query(f"""
        SELECT COUNT(*) AS total, COUNT(e.issue_id) AS indexed
        FROM `{settings.databricks_catalog}`.`{settings.databricks_schema}`.github_issues i
        LEFT JOIN (SELECT DISTINCT issue_id FROM {_index_table()}) e
          ON i.issue_id = e.issue_id
    """)[0]
    total, indexed = row["total"], row["indexed"]
    return total > 0 and indexed >= max(1, (total * 99 + 99) // 100)

def vector_search_issues(vector: list[float], limit: int = 8) -> list[dict]:
    """Exact full-table cosine search over normalized precomputed embeddings.

    The generic ARRAY SQL expression supports warehouses without the newer
    vector_cosine_similarity function. A vector index service is not required.
    """
    import json

    catalog = settings.databricks_catalog
    schema = settings.databricks_schema
    # Both stored vectors and the query vector are normalized to unit length.
    # Dot product therefore gives cosine similarity.
    similarity = """
        aggregate(
            zip_with(e.embedding, from_json(?, 'ARRAY<FLOAT>'),
                     (x, y) -> CAST(x AS DOUBLE) * CAST(y AS DOUBLE)),
            CAST(0.0 AS DOUBLE),
            (acc, v) -> acc + v
        )
    """
    query = f"""
        SELECT i.issue_id, i.issue_number, i.title, i.body, i.state,
               i.url, i.updated_at, {similarity} AS similarity
        FROM {_index_table()} e
        INNER JOIN `{catalog}`.`{schema}`.`github_issues` i
          ON i.issue_id = e.issue_id
        WHERE size(e.embedding) = ?
        ORDER BY similarity DESC
        LIMIT {min(max(limit, 1), 20)}
    """
    return _query(query, [json.dumps(vector, separators=(",", ":")), len(vector)])


def issue_comments_batch(issue_ids: list[int], limit: int = 12) -> dict[int, list[dict]]:
    if not issue_ids:
        return {}
    catalog, schema = settings.databricks_catalog, settings.databricks_schema
    markers = ", ".join("?" for _ in issue_ids)
    rows = _query(f"""
        SELECT issue_id, body, author_login, created_at, url
        FROM (
            SELECT issue_id, body, author_login, created_at, url,
                   ROW_NUMBER() OVER (PARTITION BY issue_id ORDER BY created_at ASC, comment_id ASC) AS rn
            FROM `{catalog}`.`{schema}`.github_issue_comments
            WHERE issue_id IN ({markers})
        ) ranked
        WHERE rn <= {min(max(limit, 1), 100)}
        ORDER BY issue_id, rn
    """, list(issue_ids))
    grouped = {issue_id: [] for issue_id in issue_ids}
    for row in rows:
        grouped[row["issue_id"]].append(row)
    return grouped
