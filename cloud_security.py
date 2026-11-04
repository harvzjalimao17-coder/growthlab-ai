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

    if secrets.compare_digest(token, expected):
        request.state.growthlab_role = "admin"
        return await call_next(request)

    if path not in CHAT_PATHS:
        return JSONResponse(
            {"detail": "Administrator access required"},
            status_code=403,
        )

    try:
        user = verify_supabase_user(token)

        request.state.growthlab_role = "user"
        request.state.growthlab_user_id = user["id"]

        consume_chat_request(user["id"], daily_limit=20)

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
