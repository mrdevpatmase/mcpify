import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    # Nullable at the DB level (not NOT NULL) on purpose: existing rows
    # created before this field existed have nothing to backfill it with,
    # and an ALTER TABLE ... ADD COLUMN NOT NULL against a non-empty table
    # needs a default or it fails outright. Required-ness for NEW
    # signups is enforced at the API layer instead (SignupRequest).
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class Ad(Base):
    """
    This table already has live data (created before media_data/
    media_content_type existed), so - same as User's first_name/
    last_name - those two columns are nullable at the DB level and
    added via an explicit ALTER TABLE in app/db.py's init_db(), not
    just create_all (which never alters an existing table).
    """
    __tablename__ = "ads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    # Legacy: an externally-hosted image URL, from before direct upload
    # existed. Kept for backward compat with ads created that way; new
    # ads use media_data/media_content_type instead.
    image_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    link_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    impressions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    clicks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # An uploaded image or short (5-10s) video, stored directly in
    # Postgres rather than a file store/object storage - simplest option
    # that doesn't need Railway Volumes or an S3-style account set up,
    # workable because ad creatives are small and few (an admin-managed
    # handful of ads, not user-generated uploads at scale).
    media_data: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    media_content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Which page/slot this ad shows in - see app/ads_routes.py's
    # AD_PLACEMENTS for the fixed list. NOT NULL with a DB-level DEFAULT
    # (not just a Python-side one) so the ALTER TABLE that adds this to
    # the already-live ads table can backfill the existing "Aikart" ad
    # row in the same statement, unlike first_name/last_name/media_*
    # above which had no sensible default to backfill with.
    placement: Mapped[str] = mapped_column(String(50), nullable=False, server_default="landing_top")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
