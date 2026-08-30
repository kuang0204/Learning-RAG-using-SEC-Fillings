-- filings_rag schema. Apply once against an empty database:
--     psql "$FR_DSN" -f sql/001_schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    -- Position in corpus order (0..n-1). NOT identity - chunk_id is identity. Two
    -- reasons this column exists:
    --   1. gold_set_60_v3.jsonl addresses evidence by integer position in
    --      `gold_chunks`. Without ordinal, no query result could be scored against
    --      the existing gold set or against results/*.json.
    --   2. Four chunks share a byte-identical embedding (the duplicated auditor-report
    --      captions in APPLE / ALPHABET / META / PALANTIR), so exact distance ties are
    --      real, and they reach the dense top-50 for 3 of the 60 gold questions. numpy
    --      breaks those ties by position; Postgres would break them arbitrarily.
    --      ORDER BY embedding <=> q, ordinal makes the two backends agree.
    -- Re-chunking invalidates ordinal but not chunk_id. Reload rewrites it.
    ordinal        INT  NOT NULL UNIQUE,
    filing_key     TEXT NOT NULL,
    company        TEXT NOT NULL,
    fiscal_period  TEXT NOT NULL,
    part           TEXT,
    item           TEXT,
    section        TEXT,
    element_type   TEXT,
    token_count    INT,
    text           TEXT NOT NULL,
    embedding      vector(384)
);

CREATE INDEX IF NOT EXISTS chunks_company_period_idx ON chunks (company, fiscal_period);

-- DELIBERATELY NO HNSW / IVFFLAT INDEX.
-- At 3,996 rows an exact scan is sub-millisecond, so an approximate index would buy no
-- measurable latency while making Postgres results diverge from the numpy baseline that
-- every archived number in results/ was produced against. Approximate recall is a
-- silent, query-dependent error - precisely what the migration is meant to rule out.
-- Revisit at ~100k rows, and when you do, re-run scripts/verify_migration.py first to
-- record the exact-scan baseline, then measure what the index costs in recall.
--
--   CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);   -- NOT YET
