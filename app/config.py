from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
    azure_openai_embedding_deployment: str
    azure_openai_api_version: str = "2024-10-21"
    databricks_server_hostname: str
    databricks_http_path: str
    databricks_token: str
    databricks_catalog: str = "dbw_opsforge"
    databricks_schema: str = "opsforge_monitoring"
    cors_origins: str = "http://localhost:5173,http://localhost:8000"

settings = Settings()
