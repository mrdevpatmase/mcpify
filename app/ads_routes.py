from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from app.db import get_db, get_session_factory
from app.models import Ad
from app.rate_limit import limiter
from app.security import is_public_url, verify_admin_key

router = APIRouter(tags=["Ads"])

# Simplified down to exactly 3 locations, all on the landing page
# (index.html) - no more per-page slots on login/signup/dashboard.
# - top: floating popup overlapping the navbar, centered. Rotates
#   through every active ad (GET /ads/rotation), 5s each, looping.
#   Persistent - no close button, keeps rotating for the whole page
#   view regardless.
# - bottom_right: floating widget fixed at the bottom-right corner.
#   Same as top: rotates, persistent, no close button.
# - center_modal: a big centered modal with a blocking backdrop - shown
#   on page load, does NOT rotate (one ad, the most recent active one,
#   via GET /ads/current), and CAN be dismissed via its own close
#   button (clicking the backdrop does nothing) - the point of this one
#   specifically is the visitor has to consciously close it before
#   using the site, unlike top/bottom_right which never go away.
AD_PLACEMENTS = ["top", "bottom_right", "center_modal"]
DEFAULT_PLACEMENT = "top"


def _validate_placement(placement: str) -> str:
    if placement not in AD_PLACEMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid placement '{placement}'. Must be one of: {', '.join(AD_PLACEMENTS)}.",
        )
    return placement

# 15MB covers a 5-10s video at a reasonable bitrate comfortably and is
# generous for a static image; there's no server-side duration check
# (would need ffprobe/ffmpeg as a new system dependency) - "5-10
# seconds" is enforced as a size cap + admin honor system, not a
# measured video length.
MAX_MEDIA_BYTES = 15 * 1024 * 1024
ALLOWED_MEDIA_PREFIXES = ("image/", "video/")


async def _read_media(media: Optional[UploadFile]) -> Tuple[Optional[bytes], Optional[str]]:
    if media is None or not media.filename:
        return None, None
    content_type = media.content_type or ""
    if not content_type.startswith(ALLOWED_MEDIA_PREFIXES):
        raise HTTPException(status_code=400, detail="Only image or video files are allowed.")
    data = await media.read()
    if len(data) > MAX_MEDIA_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large - max {MAX_MEDIA_BYTES // (1024 * 1024)}MB.")
    return data, content_type


class UpdateAdRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    link_url: Optional[str] = Field(None, min_length=1, max_length=1000)
    image_url: Optional[str] = Field(None, max_length=1000)
    description: Optional[str] = Field(None, max_length=2000)
    active: Optional[bool] = None
    placement: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


def _serialize(ad: Ad) -> dict:
    return {
        "id": ad.id,
        "title": ad.title,
        # image_url: legacy external-link path, kept for ads created
        # before upload existed. media_url: the new upload path, served
        # from this ad's own row via GET /ads/{id}/media - never the
        # raw bytes themselves in this JSON.
        "image_url": ad.image_url,
        "media_url": f"/ads/{ad.id}/media" if ad.media_data else None,
        "media_content_type": ad.media_content_type,
        "link_url": ad.link_url,
        "description": ad.description,
        "active": ad.active,
        "placement": ad.placement,
        "start_date": ad.start_date.isoformat() if ad.start_date else None,
        "end_date": ad.end_date.isoformat() if ad.end_date else None,
        "impressions": ad.impressions,
        "clicks": ad.clicks,
        "created_at": ad.created_at.isoformat(),
    }


# ---------------------------------------------------------
# Admin: create/list/update/delete ads
# ---------------------------------------------------------

def _parse_form_datetime(value: Optional[str]) -> Optional[datetime]:
    """
    Parses an HTML <input type="datetime-local"> value ("2026-10-01T14:30",
    no timezone info - the browser gives local wall-clock time with no
    offset). Treated as UTC directly rather than converted from the
    admin's browser timezone - simpler and unambiguous (the admin panel
    labels these fields "(UTC)"), at the cost of the admin having to
    think in UTC rather than their own timezone.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date/time: '{value}'.")


@router.post("/admin/ads", summary="Create an Ad")
@limiter.limit("20/minute")
async def create_ad(
    request: Request,
    title: str = Form(..., min_length=1, max_length=200),
    link_url: str = Form(..., min_length=1, max_length=1000),
    description: Optional[str] = Form(None, max_length=2000),
    placement: str = Form(DEFAULT_PLACEMENT),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    media: Optional[UploadFile] = File(None, description="An image or a short (5-10s) video."),
    _admin: Optional[str] = Depends(verify_admin_key),
    db=Depends(get_db),
):
    # link_url is where real visitors get redirected on click - the same
    # SSRF concern as any other user-influenced outbound-ish URL in this
    # app applies (an admin-only field is still worth guarding: an admin
    # key leak or a copy-pasted malicious link shouldn't be able to point
    # this at an internal address app.web pages might render/fetch).
    is_safe, reason = await is_public_url(link_url)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Refusing to save ad link: {reason}")

    placement = _validate_placement(placement)
    media_data, media_content_type = await _read_media(media)
    parsed_start = _parse_form_datetime(start_date)
    parsed_end = _parse_form_datetime(end_date)
    if parsed_start and parsed_end and parsed_start >= parsed_end:
        raise HTTPException(status_code=400, detail="Start date must be before end date.")

    ad = Ad(
        title=title,
        link_url=link_url,
        description=description,
        placement=placement,
        start_date=parsed_start,
        end_date=parsed_end,
        media_data=media_data,
        media_content_type=media_content_type,
    )
    db.add(ad)
    await db.commit()
    return _serialize(ad)


@router.get("/ads/{ad_id}/media", summary="Get an Ad's Uploaded Image/Video")
@limiter.limit("120/minute")
async def get_ad_media(request: Request, ad_id: str):
    session_factory = get_session_factory()
    async with session_factory() as db:
        result = await db.execute(select(Ad).where(Ad.id == ad_id))
        ad = result.scalar_one_or_none()
    if not ad or not ad.media_data:
        raise HTTPException(status_code=404, detail="No media for this ad.")
    return Response(content=ad.media_data, media_type=ad.media_content_type or "application/octet-stream")


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
    if "placement" in updates and updates["placement"]:
        updates["placement"] = _validate_placement(updates["placement"])

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

async def _active_ads_for_placement(placement: str, order_desc: bool) -> list:
    """Shared by /ads/current (most recent one) and /ads/rotation (all
    of them) - active flag + optional start/end date window, both
    checked in Python since SQLAlchemy's comparison against a
    Python-side `now` for an optional nullable column is simpler this
    way than building a NULL-aware SQL WHERE clause for it."""
    now = datetime.now(timezone.utc)
    session_factory = get_session_factory()
    async with session_factory() as db:
        query = select(Ad).where(Ad.active.is_(True), Ad.placement == placement)
        query = query.order_by(Ad.created_at.desc() if order_desc else Ad.created_at.asc())
        result = await db.execute(query)
        ads = result.scalars().all()

    valid = []
    for ad in ads:
        if ad.start_date and ad.start_date > now:
            continue
        if ad.end_date and ad.end_date < now:
            continue
        valid.append(ad)
    return valid


@router.get("/ads/current", summary="Get the Current Active Ad")
@limiter.limit("60/minute")
async def get_current_ad(request: Request, placement: str = Query(DEFAULT_PLACEMENT)):
    """Public, unauthenticated - this is what each single-ad page slot
    calls (with its own ?placement=...) to decide what to render.
    Picks the most recently created ad for that placement that's active
    and within its date window (if one is set); {} if none."""
    ads = await _active_ads_for_placement(placement, order_desc=True)
    if ads:
        return {"ad": _serialize(ads[0])}
    return {"ad": None}


@router.get("/ads/rotation", summary="Get All Active Ads for a Rotating Placement")
@limiter.limit("60/minute")
async def get_ads_rotation(request: Request, placement: str = Query(DEFAULT_PLACEMENT)):
    """Public, unauthenticated - used by the two rotating placements
    (top_banner, landing_top): returns every active ad for that
    placement, oldest first, so the frontend can cycle through all of
    them (one every 5s, looping) instead of only ever showing the
    single most-recent one like /ads/current does."""
    ads = await _active_ads_for_placement(placement, order_desc=False)
    return {"ads": [_serialize(a) for a in ads], "total": len(ads)}


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
