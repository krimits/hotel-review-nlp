"""FastAPI router for analytics and recommendations endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from reviewnlp.analytics.recommendations import build_analytics
from reviewnlp.analytics.store import AspectStore, get_aspect_store
from reviewnlp.serving.absa_schemas import AspectAnalytics, RecommendationResponse
from reviewnlp.serving.security import authorize_hotel

router = APIRouter(tags=["Analytics"])

NO_STORE_DETAIL = (
    "No aspect store is configured. Set REVIEWNLP_DB_PATH to enable "
    "hotel-scoped persistence and recommendations; no example figures are returned."
)


@router.get(
    "/hotels/{hotel_id}/recommendations",
    response_model=RecommendationResponse,
    responses={501: {"description": "No aspect store is configured"}},
)
def get_recommendations(
    hotel_id: str,
    store: Annotated[AspectStore | None, Depends(get_aspect_store)],
    days: int = Query(default=30, ge=7, le=365, description="Number of days to analyze"),
    x_api_key: str | None = Header(default=None),
) -> RecommendationResponse:
    """Aspects for one hotel, worst first, with a recommendation for each.

    Every figure returned is computed from the stored aspect rows for this
    hotel and window. When no store is configured the endpoint fails with 501
    instead of answering with examples: a caller cannot tell an invented number
    from a measured one, so it must never receive one.

    Args:
        hotel_id: Unique hotel identifier
        store: Aspect row source; None when no persistence layer is wired up
        days: Number of days to look back (default: 30, min: 7, max: 365)

    Returns:
        Aspect analytics sorted by priority score, highest first

    Raises:
        HTTPException: 501 when there is no store to read from
    """
    authorize_hotel(hotel_id, x_api_key)
    if store is None:
        raise HTTPException(status_code=501, detail=NO_STORE_DETAIL)

    analytics = build_analytics(
        store.aspect_rows(hotel_id, days),
        store.previous_period_rows(hotel_id, days),
    )
    # The action list is for problems to fix. Positive-only topics do not
    # become "complaints" merely because many guests mentioned them.
    analytics = [item for item in analytics if item["negative_count"] > 0]
    return RecommendationResponse(
        hotel_id=hotel_id,
        period_days=days,
        recommendations=[AspectAnalytics(**item) for item in analytics],
    )


@router.delete("/hotels/{hotel_id}/reviews/{review_id}", status_code=204)
def delete_review(
    hotel_id: str,
    review_id: str,
    store: Annotated[AspectStore | None, Depends(get_aspect_store)],
    source: str = Query(default="api"),
    x_api_key: str | None = Header(default=None),
) -> None:
    """Remove a review and all extracted quotes, scoped by hotel and source."""
    authorize_hotel(hotel_id, x_api_key)
    if store is None:
        raise HTTPException(status_code=501, detail=NO_STORE_DETAIL)
    if not store.delete_review(hotel_id, source, review_id):
        raise HTTPException(status_code=404, detail="review not found for this hotel and source")
