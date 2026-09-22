-- PostgreSQL schema for ABSA product
-- This schema supports multi-tenant hotel review analysis with full audit trail

-- Hotel reviews table: stores raw review data
CREATE TABLE IF NOT EXISTS hotel_reviews (
    id BIGSERIAL PRIMARY KEY,
    hotel_id VARCHAR(100) NOT NULL,
    external_review_id VARCHAR(200),
    source VARCHAR(100),
    language VARCHAR(10) NOT NULL,
    review_text TEXT NOT NULL,
    review_date TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Audit/provenance fields for reproducibility
    model_version VARCHAR(50),
    adapter_version VARCHAR(50),
    prompt_version VARCHAR(50),
    code_sha256 CHAR(64),
    
    -- Uniqueness constraint to prevent duplicate processing
    UNIQUE(hotel_id, source, external_review_id)
);

-- Index for efficient hotel-specific queries with date ranges
CREATE INDEX IF NOT EXISTS idx_reviews_hotel_date
    ON hotel_reviews(hotel_id, review_date);

-- Index for source-based filtering
CREATE INDEX IF NOT EXISTS idx_reviews_source
    ON hotel_reviews(source);

-- Review aspects table: stores extracted aspect-sentiment pairs
CREATE TABLE IF NOT EXISTS review_aspects (
    id BIGSERIAL PRIMARY KEY,
    review_id BIGINT NOT NULL REFERENCES hotel_reviews(id) ON DELETE CASCADE,
    aspect VARCHAR(50) NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    quote TEXT,
    confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for aspect-level analytics
CREATE INDEX IF NOT EXISTS idx_review_aspects_aspect_sentiment
    ON review_aspects(aspect, sentiment);

-- Index for hotel-specific aspect queries
CREATE INDEX IF NOT EXISTS idx_review_aspects_review_id
    ON review_aspects(review_id);

-- Composite index for analytics queries
CREATE INDEX IF NOT EXISTS idx_aspects_hotel_aspect
    ON review_aspects USING BTREE (review_id, aspect, sentiment);

-- Quality metrics table: tracks model performance and data quality
CREATE TABLE IF NOT EXISTS extraction_quality (
    id BIGSERIAL PRIMARY KEY,
    review_id BIGINT NOT NULL REFERENCES hotel_reviews(id) ON DELETE CASCADE,
    json_valid BOOLEAN NOT NULL,
    salvaged BOOLEAN NOT NULL,
    entries_total INTEGER,
    entries_kept INTEGER,
    entries_dropped INTEGER,
    quote_absent INTEGER,
    quote_not_in_review INTEGER,
    quote_truncated INTEGER,
    empty_valid BOOLEAN,
    generation_hit_token_budget BOOLEAN,
    processing_time_ms REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for quality monitoring
CREATE INDEX IF NOT EXISTS idx_quality_review_id
    ON extraction_quality(review_id);

-- View for aspect analytics (30-day rolling window example)
CREATE OR REPLACE VIEW aspect_analytics_30d AS
SELECT
    hr.hotel_id,
    ra.aspect,
    COUNT(*) AS mention_count,
    COUNT(*) FILTER (WHERE ra.sentiment = 'positive') AS positive_count,
    COUNT(*) FILTER (WHERE ra.sentiment = 'negative') AS negative_count,
    COUNT(*) FILTER (WHERE ra.sentiment = 'neutral') AS neutral_count,
    COUNT(*) FILTER (WHERE ra.sentiment = 'negative')::float
        / NULLIF(COUNT(*), 0) AS negative_rate,
    MIN(hr.review_date) AS first_review,
    MAX(hr.review_date) AS last_review
FROM review_aspects ra
JOIN hotel_reviews hr ON hr.id = ra.review_id
WHERE hr.review_date >= NOW() - INTERVAL '30 days'
GROUP BY hr.hotel_id, ra.aspect;

-- Function to calculate trend (comparison between periods)
CREATE OR REPLACE FUNCTION calculate_aspect_trend(
    p_hotel_id VARCHAR,
    p_aspect VARCHAR,
    p_days INTEGER DEFAULT 30
) RETURNS FLOAT AS $$
DECLARE
    current_negative_rate FLOAT;
    previous_negative_rate FLOAT;
BEGIN
    -- Current period negative rate
    SELECT negative_rate INTO current_negative_rate
    FROM aspect_analytics_30d
    WHERE hotel_id = p_hotel_id AND aspect = p_aspect;
    
    -- Previous period negative rate
    SELECT 
        COUNT(*) FILTER (WHERE ra.sentiment = 'negative')::float
            / NULLIF(COUNT(*), 0)
    INTO previous_negative_rate
    FROM review_aspects ra
    JOIN hotel_reviews hr ON hr.id = ra.review_id
    WHERE hr.hotel_id = p_hotel_id
      AND ra.aspect = p_aspect
      AND hr.review_date >= NOW() - INTERVAL '2 * p_days days'
      AND hr.review_date < NOW() - INTERVAL 'p_days days';
    
    RETURN COALESCE(current_negative_rate, 0) - COALESCE(previous_negative_rate, 0);
END;
$$ LANGUAGE plpgsql;
