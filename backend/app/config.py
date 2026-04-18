from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg2://nhl:nhl_password@localhost:5432/nhl_draft"
    REDIS_URL: str = "redis://localhost:6379/0"
    APP_ENV: str = "development"
    ENVIRONMENT: str = "development"   # used in structured log output
    SECRET_KEY: str = "change-me"
    LOG_LEVEL: str = "INFO"
    NHL_API_BASE_URL: str = "https://api-web.nhle.com/v1"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # ── Security ──────────────────────────────────────────────────────────────
    # Protects admin + ML training endpoints. Set a strong random value in prod.
    ADMIN_API_KEY: str = ""

    # ── Error tracking ────────────────────────────────────────────────────────
    # Sentry DSN. Leave empty to disable Sentry (local dev, CI).
    SENTRY_DSN: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = 0.2   # 20% of requests traced

    # ── Distributed tracing (OpenTelemetry) ───────────────────────────────────
    # OTLP endpoint (e.g. http://otel-collector:4317). Leave empty to disable.
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""
    OTEL_SERVICE_NAME: str = "nhl-draft-api"

    # ── AWS ───────────────────────────────────────────────────────────────────
    AWS_REGION: str = "us-east-1"
    # S3 bucket for model artifact storage. Leave empty to disable S3.
    AWS_S3_BUCKET: str = ""
    # Secrets Manager secret name. Leave empty to use plain env vars.
    AWS_SECRETS_NAME: str = ""
    # CloudWatch log group. Leave empty to disable direct CloudWatch shipping.
    AWS_CLOUDWATCH_LOG_GROUP: str = ""

    # ── Lambda / API Gateway ──────────────────────────────────────────────────
    # Endpoint for admin operations delegated to Lambda via API Gateway.
    # Leave empty to use the local FastAPI admin endpoints instead.
    LAMBDA_API_GATEWAY_URL: str = ""

    # ── AI / LLM ──────────────────────────────────────────────────────────────
    # Anthropic Claude API key. Required for "Ask the Scout" agent and draft summaries.
    ANTHROPIC_API_KEY: str = ""
    # Claude model to use for chat/summaries.
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"
    # Ollama base URL for local embeddings (nomic-embed-text, 768-dim).
    # Run: ollama pull nomic-embed-text  before starting the server.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # MCP servers for the Scout agent — JSON array of server configs.
    # Each entry: {"type": "url", "url": "https://...", "name": "server-name"}
    # Leave empty to disable MCP (uses direct tool-use only).
    # Example: '[{"type":"url","url":"https://mcp.example.com","name":"hockey-stats"}]'
    MCP_SERVERS: str = ""

    # ── SQS ───────────────────────────────────────────────────────────────────
    # SQS queue URL for async RAG index rebuild jobs.
    # When set, POST /api/agent/index enqueues a message and returns immediately.
    # A Lambda/ECS consumer calls build_index() from the queue.
    # Leave empty to fall back to synchronous rebuild (dev mode).
    SQS_INDEX_QUEUE_URL: str = ""

    # ── Read replica ─────────────────────────────────────────────────────────
    # Optional read-replica DATABASE_URL for agent tool queries.
    # All 6 Scout tools are read-only — routing them to a replica keeps the
    # primary free for writes (ingestion, training, conversation persistence).
    # Leave empty to use the primary DATABASE_URL for all queries.
    DATABASE_READ_REPLICA_URL: str = ""


# ── Secrets Manager bootstrap ─────────────────────────────────────────────────
# Runs before Settings() so that any secrets fetched from AWS are visible
# to pydantic-settings when it reads os.environ.
# No-op when AWS_SECRETS_NAME is not set (local dev, CI, etc.).
try:
    from app.aws.secrets import load_secrets_into_env
    load_secrets_into_env()
except Exception:
    pass  # boto3 not installed or AWS not configured — fall through to env vars

settings = Settings()
