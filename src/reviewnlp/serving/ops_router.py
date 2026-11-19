"""/ops: the operations demo. A review becomes a card of evidenced findings; the manager confirms them; confirmed
open problems become department tickets with a notification; the guest reply is drafted from what was done.

Experimental and unvalidated. Every route checks the hotel's API key first. It needs a database
(REVIEWNLP_DB_PATH), and either OPENROUTER_API_KEY for the model or REVIEWNLP_OPS_DEMO=1 for the demo's keyword
stand-ins; the review text goes to OpenRouter and Anthropic only in the first case.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Annotated, Literal, TypeAlias

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from reviewnlp.ops.llm import LLMError
from reviewnlp.ops.notify import Transport, deliver_due, transport_from_env
from reviewnlp.ops.service import NeedsInput, OpsService, service_from_env
from reviewnlp.ops.store import Conflict, SqliteOpsStore
from reviewnlp.serving.security import authorize_hotel

log = logging.getLogger("reviewnlp.serving")
router = APIRouter(prefix="/ops", tags=["Operations demo"])
_lock = threading.Lock()


class ReviewIn(BaseModel):
    source: str = Field(default="api", min_length=1, max_length=64)
    review_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=8000)


class ConfirmIn(BaseModel):
    still_open: bool | None = None
    room: str | None = Field(default=None, max_length=32)


class StateIn(BaseModel):
    state: Literal["assigned", "in_progress", "resolved"]


class ReplyIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@lru_cache(maxsize=4)
def _service_for(path: str) -> OpsService | None:
    return service_from_env(SqliteOpsStore(path))


def get_ops_service() -> OpsService:
    path = os.getenv("REVIEWNLP_DB_PATH")
    if not path:
        raise HTTPException(status_code=501, detail="the operations demo needs a database (REVIEWNLP_DB_PATH)")
    with _lock:
        service = _service_for(path)
    if service is None:
        raise HTTPException(status_code=503, detail="analysis is not configured: set OPENROUTER_API_KEY, "
                                                    "or REVIEWNLP_OPS_DEMO=1 for the keyword stand-in")
    return service


def get_ops_transport() -> Transport:
    return transport_from_env()


Service: TypeAlias = Annotated[OpsService, Depends(get_ops_service)]


def _run(action):
    """The service's errors as HTTP answers; the Greek message is shown to the manager as it is."""
    try:
        return action()
    except NeedsInput as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Conflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="not found") from error
    except LLMError as error:
        log.warning("ops model call failed: %s", error.kind)
        raise HTTPException(status_code=502, detail="Η ανάλυση δεν είναι διαθέσιμη. Δοκιμάστε ξανά σε λίγο.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/hotels/{hotel_id}/reviews")
def submit_review(hotel_id: str, review: ReviewIn, service: Service,
                  x_api_key: str | None = Header(default=None)) -> dict:
    """Analyse a review into a card of findings. The same review id and text again returns the stored card."""
    authorize_hotel(hotel_id, x_api_key)
    return _run(lambda: service.submit(hotel_id, review.source, review.review_id, review.text))


@router.get("/hotels/{hotel_id}/reviews/{source}/{review_id}")
def review_card(hotel_id: str, source: str, review_id: str, service: Service,
                x_api_key: str | None = Header(default=None)) -> dict:
    authorize_hotel(hotel_id, x_api_key)
    return _run(lambda: service.card(hotel_id, source, review_id))


@router.post("/hotels/{hotel_id}/findings/{finding_id}/confirm")
def confirm_finding(hotel_id: str, finding_id: int, body: ConfirmIn, service: Service,
                    transport: Annotated[Transport, Depends(get_ops_transport)],
                    x_api_key: str | None = Header(default=None)) -> dict:
    """Confirm a finding; an open problem with a measure becomes a ticket, and its notice is sent at once when the
    channel answers, or later by the worker."""
    authorize_hotel(hotel_id, x_api_key)
    card = _run(lambda: service.confirm(hotel_id, finding_id, body.still_open, body.room))
    deliver_due(service.store, transport, hotel_id=hotel_id)
    return card


@router.post("/hotels/{hotel_id}/findings/{finding_id}/reject")
def reject_finding(hotel_id: str, finding_id: int, service: Service,
                   x_api_key: str | None = Header(default=None)) -> dict:
    authorize_hotel(hotel_id, x_api_key)
    return _run(lambda: service.reject(hotel_id, finding_id))


@router.get("/hotels/{hotel_id}/tickets")
def list_tickets(hotel_id: str, service: Service, department: str | None = None, state: str | None = None,
                 x_api_key: str | None = Header(default=None)) -> list[dict]:
    """A department's queue: urgent first, oldest first."""
    authorize_hotel(hotel_id, x_api_key)
    return _run(lambda: service.tickets(hotel_id, department, state))


@router.post("/hotels/{hotel_id}/tickets/{ticket_id}/state")
def move_ticket(hotel_id: str, ticket_id: int, body: StateIn, service: Service,
                x_api_key: str | None = Header(default=None)) -> dict:
    authorize_hotel(hotel_id, x_api_key)
    return _run(lambda: service.move_ticket(hotel_id, ticket_id, body.state))


@router.get("/hotels/{hotel_id}/outbox")
def outbox(hotel_id: str, service: Service, x_api_key: str | None = Header(default=None)) -> list[dict]:
    authorize_hotel(hotel_id, x_api_key)
    return service.store.outbox(hotel_id)


@router.post("/hotels/{hotel_id}/outbox/deliver")
def deliver(hotel_id: str, service: Service, transport: Annotated[Transport, Depends(get_ops_transport)],
            x_api_key: str | None = Header(default=None)) -> dict:
    """Send this hotel's due notifications now (the worker does the same on a schedule)."""
    authorize_hotel(hotel_id, x_api_key)
    return deliver_due(service.store, transport, hotel_id=hotel_id)


@router.post("/hotels/{hotel_id}/reviews/{source}/{review_id}/reply/draft")
def draft_reply(hotel_id: str, source: str, review_id: str, service: Service,
                x_api_key: str | None = Header(default=None)) -> dict:
    authorize_hotel(hotel_id, x_api_key)
    return _run(lambda: service.draft_reply(hotel_id, source, review_id))


@router.post("/hotels/{hotel_id}/reviews/{source}/{review_id}/reply/approve")
def approve_reply(hotel_id: str, source: str, review_id: str, body: ReplyIn, service: Service,
                  x_api_key: str | None = Header(default=None)) -> dict:
    authorize_hotel(hotel_id, x_api_key)
    return _run(lambda: service.approve_reply(hotel_id, source, review_id, body.text))
