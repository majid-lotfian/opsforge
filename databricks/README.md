# Databricks ingestion

Import [ingest_github_issues.py](ingest_github_issues.py) as a Databricks notebook (Workspace > Import), or open it from a Databricks Git folder.

## Inputs
- Unity Catalog external location: `opsforge_github_issues_raw`
- Read-only ADLS Gen2 path: `abfss://github-issues-raw@opsforgestorage.dfs.core.windows.net/github/apache/airflow/issues/*/latest.json`
- Storage credential: `opsforge_storage_credential`

## Outputs
Managed Delta tables in `dbw_opsforge.opsforge_monitoring`:
- `github_issues`: one row per GitHub issue, latest snapshot
- `github_issue_comments`: one row per GitHub comment, latest snapshot

## Execution
Use a Unity Catalog-enabled compute resource and a principal with SELECT/READ FILES on the external location and CREATE TABLE, USE CATALOG, USE SCHEMA on the target schema. Managed table storage must already be configured for the catalog/schema. Run the notebook after GitHub Actions completes. Re-running is idempotent for the same snapshots. No raw blobs are changed.

## Important limitations
- No Databricks job has been deployed or executed yet; GitHub commit alone does not run a notebook.
- First run may require compute provisioning and incur Azure/Databricks charges.
- Issue snapshots are latest-only, not a historical change log.
- If a whole issue blob disappears, its previously ingested issue record is not automatically deleted.
- File event features are not used.
