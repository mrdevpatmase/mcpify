import os
from typing import AsyncGenerator, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


def _prepare_database_url(url: str) -> Tuple[str, dict]:
    """
    Returns (sqlalchemy_url, connect_args) for the asyncpg driver.

    Handles two real gaps between what managed Postgres providers (Neon,
    Railway, Heroku-style) put in DATABASE_URL and what asyncpg's
    connect() actually accepts:

    - Scheme: postgres:// / postgresql:// -> postgresql+asyncpg://, so
      SQLAlchemy's async engine picks the right driver.
    - Query params libpq-style drivers understand but asyncpg's connect()
      does NOT accept as keyword args - passing them through as-is
      raises "unexpected keyword argument". `sslmode` is translated to
      asyncpg's own `ssl` connect arg (same 'require'/'verify-full'/...
      string vocabulary); `channel_binding` (Neon includes this by
      default) has no asyncpg equivalent and is dropped - TLS is still
      enforced via `ssl`, only the extra SCRAM channel-binding layer on
      top isn't, which asyncpg doesn't implement regardless of the URL.
    - `statement_cache_size=0`: set whenever the host looks pooled
      (Neon/Supabase-style "-pooler" in the hostname, or any PgBouncer
      front-end) - a transaction-mode pooler can hand consecutive
      queries to different backend connections, and asyncpg's
      client-side prepared-statement cache then breaks with "prepared
      statement ... does not exist" errors. Harmless against a direct
      (non-pooled) connection too, so this is set defensively either way
      rather than trying to detect which one is in use.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]

    if not url:
        return url, {}

    parts = urlsplit(url)
    connect_args: dict = {"statement_cache_size": 0}

    kept_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "sslmode":
            connect_args["ssl"] = value
        elif key == "channel_binding":
            continue
        else:
            kept_pairs.append((key, value))

    clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept_pairs), parts.fragment))
    return clean_url, connect_args


DATABASE_URL, _DB_CONNECT_ARGS = _prepare_database_url(os.getenv("DATABASE_URL", ""))


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory = None


def get_engine():
    """Lazy singleton: only actually connects on first real use (a
    signup/login call, or init_db at startup), so importing this module
    - and running the test suite, which has no Postgres - never needs
    DATABASE_URL set. Raises clearly instead of a confusing SQLAlchemy
    error if a caller reaches here without it configured."""
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set - required for signup/login/proxy-ownership features."
            )
        _engine = create_async_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_DB_CONNECT_ARGS)
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: one session per request, always closed after."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


async def init_db() -> None:
    """
    Creates tables that don't exist yet. Called once at app startup
    (main.py's lifespan) - idempotent, safe to run on every boot. A
    schema-migration tool (Alembic) is the right next step once this
    schema needs to change more than occasionally; deferred until it's
    actually needed.

    create_all only creates whole NEW tables - it does nothing for a
    column added to an existing model when the table already exists in
    a live database (this one has real signed-up users already), so an
    added column needs its own explicit, idempotent ADD COLUMN here or
    every INSERT/SELECT referencing it breaks against production with
    "column does not exist" the moment the new code deploys.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name VARCHAR(100)"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name VARCHAR(100)"))
        await conn.execute(text("ALTER TABLE ads ADD COLUMN IF NOT EXISTS media_data BYTEA"))
        await conn.execute(text("ALTER TABLE ads ADD COLUMN IF NOT EXISTS media_content_type VARCHAR(100)"))
