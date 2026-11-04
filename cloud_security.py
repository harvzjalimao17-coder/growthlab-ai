import os
import secrets
import logging

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from supabase_auth import verify_supabase_user, consume_chat_request


CHAT_PATHS = {
    "/chat",
    "/ask",
    "/ask/stream",
}

ADMIN_PREFIXES = (
    "/documents",
    "/images",
)

BLOCKED_DOC_PATHS = {
    "/docs",
    "/redoc",
    "/openapi.json",
    "/docs/oauth2-redirect",
}

# This address is used only after Supabase validates the access token.
ADMIN_EMAIL = "harvzjalimao17@gmail.com"
REGULAR_USER_DAILY_LIMIT = 20


def _verified_email(user):
    """Read an email only from the trusted Supabase verification result."""
    if isinstance(user, dict):
        return str(user.get("email") or "").strip().casefold()
    return str(getattr(user, "email", "") or "").strip().casefold()


async def growthlab_cloud_security(request: Request, call_next):
    path = request.url.path

    if os.getenv("GROWTHLAB_CLOUD_MODE", "false").lower() != "true":
        return await call_next(request)

    if path in BLOCKED_DOC_PATHS:
        return JSONResponse({"detail": "Not found"}, status_code=404)

    if path == "/health":
        return JSONResponse({"status": "ok", "ai": "cloud"})

    protected = (
        path in CHAT_PATHS
        or any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in ADMIN_PREFIXES
        )
    )

    if not protected:
        return await call_next(request)

    expected = os.getenv("GROWTHLAB_DEPLOY_TOKEN", "")

    if not expected:
        return JSONResponse(
            {"detail": "Service not configured"},
            status_code=503,
        )

    supplied = request.headers.get("Authorization", "")
    scheme, _, credential = supplied.partition(" ")

    if scheme.lower() != "bearer" or not credential.strip():
        return JSONResponse(
            {"detail": "Login required"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credential.strip()

    # Preserve existing deployment-token administrator access.
    if secrets.compare_digest(token, expected):
        request.state.growthlab_role = "admin"
        return await call_next(request)

    # All other tokens must be verified by Supabase. Never trust an email
    # supplied in the request body, headers, or unverified JWT contents.
    try:
        user = verify_supabase_user(token)
        user_id = user["id"] if isinstance(user, dict) else user.id
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user account")

        request.state.growthlab_user_id = user_id
        is_admin = secrets.compare_digest(_verified_email(user), ADMIN_EMAIL)

        if is_admin:
            request.state.growthlab_role = "admin"
            # No application-level daily limit for the verified admin.
            # Provider-side rate limits and quotas still apply.
        else:
            request.state.growthlab_role = "user"
            if path not in CHAT_PATHS:
                return JSONResponse(
                    {"detail": "Administrator access required"},
                    status_code=403,
                )
            consume_chat_request(user_id, daily_limit=REGULAR_USER_DAILY_LIMIT)

    except HTTPException as exc:
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
        )
    except Exception:
        logging.exception("GrowthLab AI authentication failed")
        return JSONResponse(
            {"detail": "Authentication service unavailable"},
            status_code=503,
        )

    return await call_next(request)
