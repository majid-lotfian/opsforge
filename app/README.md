# OpsForge application

This is the initial integrated incident investigation application: FastAPI, Azure OpenAI, read-only Databricks SQL queries against existing Delta tables, and a browser UI. It does **not** execute infrastructure remediation.

## Azure configuration required
1. Create an Azure OpenAI / Foundry resource with a deployed chat model supporting JSON response format. Record endpoint, deployment name, API version and API key.
2. In Databricks SQL Warehouses, select a SQL warehouse and copy its HTTP Path. Provide a Databricks access token with permission to use the warehouse and SELECT on the two Delta tables.
3. Populate the settings in `.env.example` as environment variables. Never commit a populated .env file.
4. Deploy this API to an Azure App Service or another supported Python host. Startup command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`. Set the working directory to the repository root and install `app/requirements.txt`.

## Current limitations
- This version uses bounded lexical search, not vector similarity. Embedding deployment setting is reserved for the upcoming semantic retrieval integration.
- Databricks SQL warehouse usage and Azure OpenAI inference are billed.
- No application user authentication is implemented. Do not expose the API publicly before configuring authentication and rate limits.
- Azure/Databricks resources are configured manually, not via IaC.
