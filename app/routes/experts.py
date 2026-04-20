"""GET /experts/{candidate_id} — full candidate profile as Markdown."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.profile_builder import render_profile_for_id


router = APIRouter()


class ExpertProfileResponse(BaseModel):
    candidate_id: str
    markdown: str


@router.get("/experts/{candidate_id}", response_model=ExpertProfileResponse)
def get_expert(candidate_id: str) -> ExpertProfileResponse:
    s = get_settings()
    try:
        md = render_profile_for_id(s.database_url, candidate_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ExpertProfileResponse(candidate_id=candidate_id, markdown=md)
