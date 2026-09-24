"""Pydantic schemas for the ABSA inference and analytics API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AbsaRequest(BaseModel):
    """Request schema for ABSA analysis endpoint."""

    hotel_id: str = Field(..., min_length=1, max_length=100, description="Unique hotel identifier")
    review_id: str | None = Field(default=None, max_length=200, description="Optional external review ID")
    text: str = Field(..., min_length=3, max_length=8000, description="Review text to analyze")
    source: str | None = Field(default=None, max_length=100, description="Review source (e.g., booking.com)")
    language: Literal["en"] = Field(default="en", description="English ABSA only; Greek hotel aspects have not been evaluated")
    review_date: datetime | None = Field(default=None, description="Date of the review")


class AbsaAspect(BaseModel):
    """Single aspect extracted from a review."""

    aspect: str = Field(..., description="Aspect category (cleanliness, staff, etc.)")
    sentiment: str = Field(..., description="Sentiment: positive, negative, or neutral")
    quote: str = Field(..., description="Exact quote from the review supporting this aspect")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Optional confidence score")


class AbsaResponse(BaseModel):
    """Response schema for ABSA analysis endpoint."""

    hotel_id: str
    review_id: str | None
    stored: bool = Field(default=False, description="Whether this review was saved for hotel analytics")
    aspects: list[AbsaAspect]
    overall_sentiment: str | None
    json_valid: bool
    salvaged: bool
    entries_dropped: int
    quote_absent: int
    quote_not_in_review: int
    generation_hit_token_budget: bool
    processing_time_ms: float | None = Field(
        default=None,
        description=(
            "Wall time for this review, when it was generated on its own. "
            "Null inside a batch: the batch is one padded forward pass, so no "
            "per-review share of it was measured - see total_processing_time_ms."
        ),
    )


class AspectAnalytics(BaseModel):
    """Aggregated analytics for a single aspect."""

    aspect: str
    review_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    negative_rate: float
    trend: float | None
    priority_score: float
    recommendation: str


class RecommendationResponse(BaseModel):
    """Response schema for recommendations endpoint."""

    hotel_id: str
    period_days: int
    recommendations: list[AspectAnalytics]


class EvidenceExample(BaseModel):
    source: str
    review_id: str
    review_date: datetime
    quote: str


class EvidenceResponse(BaseModel):
    hotel_id: str
    aspect: str
    period_days: int
    examples: list[EvidenceExample]


class BatchAbsaRequest(BaseModel):
    """Request schema for batch ABSA analysis."""

    hotel_id: str = Field(..., min_length=1, max_length=100)
    reviews: list[AbsaRequest] = Field(..., min_length=1, max_length=256)


class BatchAbsaResponse(BaseModel):
    """Response schema for batch ABSA analysis."""

    hotel_id: str
    results: list[AbsaResponse]
    total_processing_time_ms: float
