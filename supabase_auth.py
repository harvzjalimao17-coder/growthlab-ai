import os
import requests
from fastapi import HTTPException


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.getenv(
    "SUPABASE_PUBLISHABLE_KEY", ""
)
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY", ""
)


def verify_supabase_user(access_token: str) -> dict:
    """Verify a Supabase access token with Supabase Auth."""

    if not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
        raise HTTPException(
            status_code=503,
            detail="Supabase authentication is not configured.",
        )

    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_PUBLISHABLE_KEY,
                "Authorization": f"Bearer {access_token}",
            },
            timeout=15,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable.",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired login session.",
        )

    user = response.json()

    if not user.get("id"):
        raise HTTPException(
            status_code=401,
            detail="Invalid user identity.",
        )

    return user


def consume_chat_request(user_id: str, daily_limit: int = 20) -> dict:
    """Atomically consume one daily chat request."""

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI usage tracking is not configured.",
        )

    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/gl_ai_consume_chat_request",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": (
                    f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"
                ),
                "Content-Type": "application/json",
            },
            json={
                "p_user_id": user_id,
                "p_daily_limit": daily_limit,
            },
            timeout=15,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="AI usage tracking is temporarily unavailable.",
        )

    if not response.ok:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify daily AI usage.",
        )

    result = response.json()

    if isinstance(result, list):
        result = result[0] if result else {}

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=503,
            detail="Invalid usage tracking response.",
        )

    if result.get("allowed") is not True:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily AI limit reached "
                f"({daily_limit} requests). "
                "Please try again tomorrow."
            ),
        )

    return result
