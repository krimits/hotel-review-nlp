"""FastAPI router for analytics and recommendations endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from reviewnlp.serving.absa_schemas import (
    AspectAnalytics,
    RecommendationResponse,
)
from reviewnlp.analytics.recommendations import build_recommendation

router = APIRouter(tags=["Analytics"])


@router.get(
    "/hotels/{hotel_id}/recommendations",
    response_model=RecommendationResponse,
)
def get_recommendations(
    hotel_id: str,
    days: int = Query(default=30, ge=7, le=365, description="Number of days to analyze"),
) -> RecommendationResponse:
    """Get prioritized recommendations for a hotel based on aspect analytics.
    
    This endpoint analyzes reviews from the specified period and returns
    aspects sorted by priority score, along with actionable recommendations.
    
    In production, this would query aggregated metrics from PostgreSQL,
    calculate trends against previous periods, and return only the highest-
    priority aspects.
    
    Args:
        hotel_id: Unique hotel identifier
        days: Number of days to look back (default: 30, min: 7, max: 365)
        
    Returns:
        List of aspect analytics with recommendations, sorted by priority
    """
    # Production implementation:
    # 1. query aggregated metrics from PostgreSQL
    # 2. calculate trend against the previous period
    # 3. calculate priority_score
    # 4. return only the highest-priority aspects

    # Placeholder data for demonstration
    metrics = [
        {
            "aspect": "cleanliness",
            "review_count": 182,
            "positive_count": 91,
            "negative_count": 71,
            "neutral_count": 20,
            "negative_rate": 71 / 182,
            "trend": 0.12,
            "priority_score": 0.86,
        },
        {
            "aspect": "staff",
            "review_count": 210,
            "positive_count": 170,
            "negative_count": 25,
            "neutral_count": 15,
            "negative_rate": 25 / 210,
            "trend": -0.03,
            "priority_score": 0.29,
        },
    ]

    recommendations = [
        AspectAnalytics(
            aspect=item["aspect"],
            review_count=item["review_count"],
            positive_count=item["positive_count"],
            negative_count=item["negative_count"],
            neutral_count=item["neutral_count"],
            negative_rate=round(item["negative_rate"], 4),
            trend=item["trend"],
            priority_score=item["priority_score"],
            recommendation=build_recommendation(item["aspect"]),
        )
        for item in sorted(
            metrics,
            key=lambda item: item["priority_score"],
            reverse=True,
        )
    ]

    return RecommendationResponse(
        hotel_id=hotel_id,
        period_days=days,
        recommendations=recommendations,
    )
