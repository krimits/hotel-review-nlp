# ABSA Product Implementation Guide

This document records the original product design. For the implemented local
pilot, API keys, SQLite storage, dashboard and current evaluation requirements,
see [docs/PRODUCT_SETUP.md](../docs/PRODUCT_SETUP.md). This file's SQL schema
is a PostgreSQL-oriented design reference; the executable pilot schema lives
in `reviewnlp.analytics.store`. Neither is an evaluated ABSA production claim.

## Architecture Overview

```
POST /absa → ABSA Model → Aspect Records → Database → Analytics → Recommendations
```

## New Files Created

### 1. API Schemas (`src/reviewnlp/serving/absa_schemas.py`)
- `AbsaRequest`: Input schema with hotel_id, review text, metadata
- `AbsaResponse`: Output with aspects, sentiments, quotes, quality flags
- `AspectAnalytics`: Aggregated metrics per aspect
- `RecommendationResponse`: Prioritized recommendations

### 2. Model Manager (`src/reviewnlp/serving/absa_model_manager.py`)
- Singleton pattern: loads model once at startup (not per request)
- Thread-safe with locking
- Provides `analyze()` and `analyze_batch()` methods

### 3. ABSA Router (`src/reviewnlp/serving/absa_router.py`)
- `POST /absa`: Single review analysis
- `POST /absa/batch`: Batch processing for multiple reviews

### 4. Analytics Router (`src/reviewnlp/serving/analytics_router.py`)
- `GET /hotels/{hotel_id}/recommendations`: Priority recommendations

### 5. Recommendation Engine (`src/reviewnlp/analytics/recommendations.py`)
- Priority scoring formula: 45% negative_rate + 25% volume + 20% trend + 10% business_weight
- Predefined Greek recommendations per aspect category

### 6. Database Schema (`scripts/schema.sql`)
- `hotel_reviews`: Raw review storage with audit fields
- `review_aspects`: Extracted aspect-sentiment pairs
- `extraction_quality`: Quality metrics for monitoring
- Indexes for efficient analytics queries
- View for 30-day rolling analytics

## API Endpoints

### Analyze Review
```bash
curl -X POST http://localhost:8000/absa \
  -H "Content-Type: application/json" \
  -d '{
    "hotel_id": "hotel-athens-001",
    "review_id": "booking-98765",
    "source": "booking.com",
    "language": "en",
    "text": "The room was clean and spacious, but breakfast was poor."
  }'
```

### Get Recommendations
```bash
curl "http://localhost:8000/hotels/hotel-athens-001/recommendations?days=30"
```

## Key Production Improvements

1. **Model Loading**: Model loaded once at startup via `AbsaModelManager`, not per request
2. **Hotel ID Tracking**: All data linked to `hotel_id` for multi-tenant support
3. **Quality Flags**: `json_valid`, `salvaged`, `entries_dropped` tracked for monitoring
4. **Audit Trail**: Model version, adapter version, code SHA256 stored for reproducibility
5. **Idempotency**: Unique constraint on `(hotel_id, source, external_review_id)`
6. **Priority Scoring**: Confidence-adjusted scoring considering volume, not just rates

## Deployment

```bash
# Set environment variables
export ABSA_ADAPTER_DIR=/path/to/adapter  # optional
export ABSA_DEVICE=cuda  # or cpu

# Run the API
uvicorn reviewnlp.serving.app:app --host 0.0.0.0 --port 8000
```

## Monitoring Metrics

Track these Prometheus-style metrics:
- `absa_requests_total`: Total requests
- `absa_latency_seconds`: Processing time
- `absa_json_valid_rate`: JSON parsing success rate
- `absa_salvaged_rate`: Rate of salvaged outputs
- `absa_entries_dropped_rate`: Entry rejection rate
- `absa_model_errors_total`: Model inference errors

## Next Steps (Phase 2+)

1. Evaluate ABSA against independently labeled hotel aspects
2. Move the SQLite pilot to managed storage and managed authentication for multiple workers
3. Implement PII redaction for quotes
4. Add batch ingestion from CSV/API providers
5. Build dashboard with trends and alerts
6. Connect to operational KPIs (rating, occupancy)
