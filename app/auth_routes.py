from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import create_access_token, get_current_user, hash_password, verify_password
from app.db import get_db
from app.models import User
from app.rate_limit import limiter

router = APIRouter(prefix="/auth", tags=["Auth"])


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/signup", response_model=TokenResponse, summary="Create an account")
@limiter.limit("10/minute")
async def signup(request: Request, payload: SignupRequest, db: AsyncSession = Depends(get_db)):
    # Normalize case so "Foo@Bar.com" and "foo@bar.com" are the same
    # account - EmailStr doesn't do this, and Postgres's unique index on
    # email is case-sensitive by default. func.lower() on the lookup
    # side also catches any legacy row stored with mixed case before
    # this normalization existed.
    normalized_email = payload.email.strip().lower()
    existing = await db.execute(select(User).where(func.lower(User.email) == normalized_email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user = User(
        email=normalized_email,
        hashed_password=hash_password(payload.password),
        first_name=payload.first_name,
        last_name=payload.last_name,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        # The existence check above and this insert aren't atomic - two
        # concurrent signups for the same email can both pass the check
        # and race to insert. The unique constraint on email correctly
        # stops the second one at the DB level, but without this handler
        # that surfaced as a raw 500 instead of the same clean 409 the
        # non-race duplicate case already returns.
        await db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse, summary="Log in")
@limiter.limit("10/minute")
async def login(request: Request, payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(func.lower(User.email) == payload.email.strip().lower()))
    user = result.scalar_one_or_none()
    # Same error for "no such user" and "wrong password" - distinguishing
    # them lets a caller enumerate which emails have accounts.
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me", summary="Current logged-in user")
async def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "created_at": current_user.created_at.isoformat(),
    }
