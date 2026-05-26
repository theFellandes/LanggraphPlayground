-- Run automatically on first container start by docker-entrypoint-initdb.d.
-- Idempotent — safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;

-- langchain-postgres creates its own tables on first use; this file just
-- guarantees the extension is loaded before LangChain's first connection.
-- The 384 here matches BAAI/bge-small-en-v1.5; change it if you swap embedders.
