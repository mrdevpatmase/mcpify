import asyncio
import io
import os
from typing import Tuple


def is_configured() -> bool:
    """
    True once CLOUDINARY_CLOUD_NAME/API_KEY/API_SECRET are all set. Ad
    uploads fall back to storing bytes directly in Postgres (the old
    behavior) when this is False, so local dev / a deployment that
    hasn't set these up yet never breaks ad creation outright - it just
    doesn't get the egress-saving benefit until they're configured.
    """
    return bool(
        os.getenv("CLOUDINARY_CLOUD_NAME")
        and os.getenv("CLOUDINARY_API_KEY")
        and os.getenv("CLOUDINARY_API_SECRET")
    )


def _configure() -> None:
    import cloudinary

    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )


async def upload_media(data: bytes, content_type: str) -> Tuple[str, str]:
    """
    Uploads ad media (image or short video) to Cloudinary, returns
    (secure_url, public_id). public_id is stored on the Ad row so the
    asset can be cleaned up later (delete_media) when the ad is deleted
    or its media is replaced - without it, every edit/delete would
    leave an orphaned file sitting in the Cloudinary account forever.

    Run in a thread: the cloudinary SDK's uploader is a synchronous,
    blocking HTTP call (no asyncio support), same reasoning as
    app/analytics.py's GA4 client.
    """
    import cloudinary.uploader

    _configure()
    resource_type = "video" if (content_type or "").startswith("video/") else "image"
    result = await asyncio.to_thread(
        cloudinary.uploader.upload,
        io.BytesIO(data),
        resource_type=resource_type,
        folder="mcpify_ads",
    )
    return result["secure_url"], result["public_id"]


async def delete_media(public_id: str, content_type: str) -> None:
    """Best-effort cleanup - a failure here (network blip, already-deleted
    asset) shouldn't block the ad delete/replace it's cleaning up after,
    so callers should swallow exceptions from this rather than propagate
    a 500 for what's just a storage-quota housekeeping step."""
    import cloudinary.uploader

    _configure()
    resource_type = "video" if (content_type or "").startswith("video/") else "image"
    await asyncio.to_thread(cloudinary.uploader.destroy, public_id, resource_type=resource_type)
