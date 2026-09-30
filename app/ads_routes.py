from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from app.db import get_db, get_session_factory
from app.models import Ad
from app.rate_limit import limiter
from app.security import is_public_url, verify_admin_key

router = APIRouter(tags=["Ads"])


class CreateAdRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    link_url: str = Field(..., min_length=1, max_length=1000)
    image_url: Optional[str] = Field(None, max_length=1000)
    description: Optional[str] = Field(None, max_length=2000)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class UpdateAdRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    link_url: Optional[str] = Field(None, min_length=1, max_length=1000)
    image_url: Optional[str] = Field(None, max_length=1000)
    description: Optional[str] = Field(None, max_length=2000)
    active: Optional[bool] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


def _serialize(ad: Ad) -> dict:
    return {
        "id": ad.id,
        "title": ad.title,
        "image_url": ad.image_url,
        "link_url": ad.link_url,
        "description": ad.description,
        "active": ad.active,
        "start_date": ad.start_date.isoformat() if ad.start_date else None,
        "end_date": ad.end_date.isoformat() if ad.end_date else None,
        "impressions": ad.impressions,
        "clicks": ad.clicks,
        "created_at": ad.created_at.isoformat(),
    }


# ---------------------------------------------------------
# Admin: create/list/update/delete ads
# ---------------------------------------------------------

@router.post("/admin/ads", summary="Create an Ad")
@limiter.limit("20/minute")
async def create_ad(
    request: Request,
    payload: CreateAdRequest,
    _admin: Optional[str] = Depends(verify_admin_key),
    db=Depends(get_db),
):
    # link_url is where real visitors get redirected on click - the same
    # SSRF concern as any other user-influenced outbound-ish URL in this
    # app applies (an admin-only field is still worth guarding: an admin
    # key leak or a copy-pasted malicious link shouldn't be able to point
    # this at an internal address app.web pages might render/fetch).
    is_safe, reason = await is_public_url(payload.link_url)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Refusing to save ad link: {reason}")

    ad = Ad(
        title=payload.title,
        link_url=payload.link_url,
        image_url=payload.image_url,
        description=payload.description,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
    db.add(ad)
    await db.commit()
    return _serialize(ad)


@router.get("/admin/ads", summary="List All Ads")
@limiter.limit("30/minute")
async def list_ads(request: Request, _admin: Optional[str] = Depends(verify_admin_key), db=Depends(get_db)):
    result = await db.execute(select(Ad).order_by(Ad.created_at.desc()))
    ads = result.scalars().all()
    return {"ads": [_serialize(a) for a in ads], "total": len(ads)}


@router.patch("/admin/ads/{ad_id}", summary="Update an Ad")
@limiter.limit("20/minute")
async def update_ad(
    request: Request,
    ad_id: str,
    payload: UpdateAdRequest,
    _admin: Optional[str] = Depends(verify_admin_key),
    db=Depends(get_db),
):
    result = await db.execute(select(Ad).where(Ad.id == ad_id))
    ad = result.scalar_one_or_none()
    if not ad:
        raise HTTPException(status_code=404, detail=f"Ad '{ad_id}' not found.")

    updates = payload.model_dump(exclude_unset=True)
    if "link_url" in updates and updates["link_url"]:
        is_safe, reason = await is_public_url(updates["link_url"])
        if not is_safe:
            raise HTTPException(status_code=400, detail=f"Refusing to save ad link: {reason}")

    for field, value in updates.items():
        setattr(ad, field, value)
    await db.commit()
    return _serialize(ad)


@router.delete("/admin/ads/{ad_id}", summary="Delete an Ad")
@limiter.limit("20/minute")
async def delete_ad(request: Request, ad_id: str, _admin: Optional[str] = Depends(verify_admin_key), db=Depends(get_db)):
    result = await db.execute(select(Ad).where(Ad.id == ad_id))
    ad = result.scalar_one_or_none()
    if not ad:
        raise HTTPException(status_code=404, detail=f"Ad '{ad_id}' not found.")
    await db.delete(ad)
    await db.commit()
    return {"deleted": True, "id": ad_id}


# ---------------------------------------------------------
# Public: serve the current ad, track impressions/clicks
# ---------------------------------------------------------

@router.get("/ads/current", summary="Get the Current Active Ad")
@limiter.limit("60/minute")
async def get_current_ad(request: Request):
    """Public, unauthenticated - this is what the landing page calls to
    decide what to render. Picks the most recently created ad that's
    active and within its date window (if one is set); {} if none."""
    now = datetime.now(timezone.utc)
    session_factory = get_session_factory()
    async with session_factory() as db:
        result = await db.execute(select(Ad).where(Ad.active.is_(True)).order_by(Ad.created_at.desc()))
        ads = result.scalars().all()

    for ad in ads:
        if ad.start_date and ad.start_date > now:
            continue
        if ad.end_date and ad.end_date < now:
            continue
        return {"ad": _serialize(ad)}
    return {"ad": None}


@router.post("/ads/{ad_id}/impression", summary="Record an Ad Impression")
@limiter.limit("120/minute")
async def record_impression(request: Request, ad_id: str):
    """Fire-and-forget beacon the frontend calls once per ad render.
    No auth (anonymous visitors see ads before ever logging in) - rate
    limited instead to bound abuse."""
    session_factory = get_session_factory()
    async with session_factory() as db:
        await db.execute(update(Ad).where(Ad.id == ad_id).values(impressions=Ad.impressions + 1))
        await db.commit()
    return {"ok": True}


@router.get("/ads/{ad_id}/click", summary="Record an Ad Click and Redirect")
@limiter.limit("120/minute")
async def record_click(request: Request, ad_id: str):
    """The ad's href points HERE, not directly at link_url - a plain
    server redirect means the click is counted even if the visitor's
    browser has JavaScript disabled or blocks the fetch beacon, and it
    works with a normal <a href> (middle-click/open-in-new-tab, no JS
    needed to navigate)."""
    session_factory = get_session_factory()
    async with session_factory() as db:
        result = await db.execute(select(Ad).where(Ad.id == ad_id))
        ad = result.scalar_one_or_none()
        if not ad:
            raise HTTPException(status_code=404, detail="Ad not found.")
        await db.execute(update(Ad).where(Ad.id == ad_id).values(clicks=Ad.clicks + 1))
        await db.commit()
        target = ad.link_url
    return RedirectResponse(url=target, status_code=302)
