from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str = "opsforge-chat"
    azure_openai_embedding_deployment: str = "opsforge-embedding"
    databricks_server_hostname: str
    databricks_http_path: str
    databricks_token: str
    databricks_write_token: str | None = None
    databricks_catalog: str = "dbw_opsforge"
    databricks_schema: str = "opsforge_monitoring"
    cors_origins: str = "http://localhost:5173,http://localhost:8000"

settings = Settings()
