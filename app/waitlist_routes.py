from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import WaitlistEntry
from app.rate_limit import limiter

router = APIRouter(tags=["Waitlist"])

VALID_TIERS = {"pro", "enterprise"}


class WaitlistJoinRequest(BaseModel):
    email: EmailStr
    tier: str


@router.post("/waitlist/join", summary="Join the Pro/Enterprise Waitlist")
@limiter.limit("10/minute")
async def join_waitlist(request: Request, payload: WaitlistJoinRequest, db: AsyncSession = Depends(get_db)):
    if payload.tier not in VALID_TIERS:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {', '.join(VALID_TIERS)}.")

    entry = WaitlistEntry(email=payload.email.strip().lower(), tier=payload.tier)
    db.add(entry)
    try:
        await db.commit()
    except IntegrityError:
        # Same (email, tier) submitted twice - not an error worth
        # surfacing to the person filling out the form, they're already
        # on the list either way.
        await db.rollback()

    return {"status": "ok", "message": "You're on the list - we'll email you when it's ready."}
