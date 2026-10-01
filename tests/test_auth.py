"""
Unit coverage for app/auth.py's pure functions (password hashing, JWT
issue/decode) and app/db.py's URL normalization - none of these need a
real Postgres connection, so they run in every environment (including
CI) without DATABASE_URL set, the same way the rest of this suite runs
Redis-free.
"""
import pytest

from app.db import _prepare_database_url


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-for-unit-tests")


def test_password_hash_round_trip():
    from app.auth import hash_password, verify_password

    hashed = hash_password("correct-password")
    assert hashed != "correct-password"
    assert verify_password("correct-password", hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_verify_password_rejects_malformed_hash():
    from app.auth import verify_password

    assert verify_password("anything", "not-a-real-bcrypt-hash") is False


def test_create_and_decode_access_token_round_trip():
    from app.auth import create_access_token, decode_access_token

    token = create_access_token("user-123")
    assert decode_access_token(token) == "user-123"


def test_decode_access_token_rejects_tampered_token():
    import jwt as pyjwt

    from app.auth import create_access_token, decode_access_token

    token = create_access_token("user-123")
    # Flips a character in the middle of the signature segment, not the
    # last character of the whole token: base64url has no padding, so a
    # JWT's trailing character only encodes a couple of real bits - for
    # some signatures, every value that trailing character could take
    # decodes to the SAME bytes, making the "flip the last char" tamper
    # a no-op roughly 1 in 4 runs (it depends on the signature's actual
    # bytes, which differ every run since the payload includes the
    # current timestamp) and failing this test nondeterministically.
    # A middle character has no such ambiguity.
    mid = len(token) // 2
    tampered = token[:mid] + ("A" if token[mid] != "A" else "B") + token[mid + 1:]
    with pytest.raises(pyjwt.PyJWTError):
        decode_access_token(tampered)


def test_decode_access_token_rejects_expired_token(monkeypatch):
    import jwt as pyjwt

    import app.auth as auth_module

    monkeypatch.setattr(auth_module, "JWT_EXPIRE_MINUTES", -1)
    token = auth_module.create_access_token("user-123")
    with pytest.raises(pyjwt.ExpiredSignatureError):
        auth_module.decode_access_token(token)


@pytest.mark.parametrize(
    "raw,expected_prefix",
    [
        ("postgres://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
        ("postgresql://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
    ],
)
def test_prepare_database_url_rewrites_scheme(raw, expected_prefix):
    clean_url, _ = _prepare_database_url(raw)
    assert clean_url == expected_prefix


def test_prepare_database_url_strips_libpq_only_params_and_translates_sslmode():
    raw = "postgres://u:p@ep-square-firefly-pooler.neon.tech/db?sslmode=require&channel_binding=require"
    clean_url, connect_args = _prepare_database_url(raw)

    assert "sslmode" not in clean_url
    assert "channel_binding" not in clean_url
    assert connect_args["ssl"] == "require"


def test_prepare_database_url_sets_statement_cache_size_zero():
    _, connect_args = _prepare_database_url("postgres://u:p@host/db")
    assert connect_args["statement_cache_size"] == 0


def test_prepare_database_url_passes_through_empty():
    clean_url, connect_args = _prepare_database_url("")
    assert clean_url == ""
    assert connect_args == {}
