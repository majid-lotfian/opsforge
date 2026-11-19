"""Read-only access to the existing Unity Catalog incident tables."""
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
