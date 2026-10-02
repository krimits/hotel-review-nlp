"""Request and result shapes of the triage workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# The five topics of the complaint pilot (docs/annotation/complaint_topics_guideline.md). They are not
# the eight ABSA aspects, and the two lists are never merged.
TOPICS = ("bathroom", "cleanliness", "air_conditioning", "pests", "responsiveness")
OTHER = "other"
DEPARTMENTS = ("reception", "housekeeping", "maintenance", "food_and_beverage", "management", "other")

Answer = Literal["yes", "no", "unsure"]
StageStatus = Literal["ok", "disabled", "error"]


class TriageRequest(BaseModel):
    hotel_id: str = Field(..., min_length=1, max_length=100, description="Unique hotel identifier")
    review_id: str | None = Field(default=None, max_length=200, description="Optional external review ID")
    text: str = Field(..., min_length=3, max_length=8000, description="The whole review, praise and complaints")
    source: str | None = Field(default=None, max_length=100, description="Review source (e.g. booking.com)")
    language: Literal["en"] = Field(default="en", description="English only; other languages have not been evaluated")
    review_date: datetime | None = Field(default=None, description="Date of the review")


class SentimentResult(BaseModel):
    label: str
    confidence: float | None = Field(description="Probability of the label; None if the model cannot give one")
    probabilities: dict[str, float] | None = Field(description="The whole distribution, when the model gives it")
    model_type: str
    model_path: str


class ComplaintTopic(BaseModel):
    topic: str = Field(description="One of the five pilot topics, or 'other' for any other complaint")
    answer: Answer
    probability: float = Field(ge=0.0, le=1.0, description="Jev's probability that the answer is 'yes'")


class ComplaintsResult(BaseModel):
    status: StageStatus
    error: str | None = Field(default=None, description="The kind of failure, never the review or a response body")
    topics: list[ComplaintTopic] = Field(default_factory=list)
    other_complaint: ComplaintTopic | None = None
    route: str | None = None
    model: str | None = Field(default=None, description="The model that answered, as the provider names it")
    questions_version: str
    questions_sha256: str


class RoutingDecision(BaseModel):
    qwen_triggered: bool
    reasons: list[str] = Field(description="Why the suggestion stage was asked for; empty when it was not")
    needs_review: bool = Field(description="A person should look at this review")
    review_reasons: list[str] = Field(default_factory=list)
    thresholds: dict[str, float | list[float]]
    thresholds_status: Literal["provisional"] = Field(
        default="provisional", description="Chosen by hand, not on a development sample")


class SuggestedAction(BaseModel):
    problem: str
    excerpt: str = Field(description="Copied from the review; the pipeline checked that it is there")
    measure: str
    department: Literal["reception", "housekeeping", "maintenance", "food_and_beverage", "management", "other"]
    to_confirm: list[str] = Field(default_factory=list, description="Facts the review does not give and the hotel must check")


class ActionsResult(BaseModel):
    status: Literal["not_triggered", "disabled", "ok", "no_grounded_actions", "error"]
    error: str | None = None
    actions: list[SuggestedAction] = Field(default_factory=list)
    dropped: int = Field(default=0, description="Generated actions rejected: malformed, ungrounded or off the list")
    model: str | None = None
    prompt_version: str | None = None


class StageTimings(BaseModel):
    sentiment_ms: float
    complaints_ms: float | None = None
    routing_ms: float
    actions_ms: float | None = None
    total_ms: float


class TriageResponse(BaseModel):
    hotel_id: str
    review_id: str | None
    stored: bool = False
    status: Literal["complete", "partial"] = Field(description="'partial' when a stage that was asked to run failed")
    sentiment: SentimentResult
    complaints: ComplaintsResult
    routing: RoutingDecision
    actions: ActionsResult
    timings: StageTimings
    validation_status: Literal["unvalidated"] = Field(
        default="unvalidated", description="No stage has been measured on whole or mixed reviews")


class TriageActionRow(BaseModel):
    review_id: str
    source: str
    review_date: datetime
    problem: str
    excerpt: str
    measure: str
    department: str
    to_confirm: list[str]


class ReviewToCheck(BaseModel):
    review_id: str
    source: str
    review_date: datetime
    reasons: list[str]


class TriageSummaryResponse(BaseModel):
    hotel_id: str
    period_days: int
    reviews: int
    sentiment: dict[str, int]
    complaints_evaluated: int = Field(description="Reviews whose complaint topics were actually asked for")
    complaints: dict[str, dict[str, int]] = Field(description="Per topic, how many reviews answered yes, no and unsure")
    actions: list[TriageActionRow]
    to_check: list[ReviewToCheck]
    validation_status: Literal["unvalidated"] = "unvalidated"
