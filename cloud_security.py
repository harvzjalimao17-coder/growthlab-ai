import os
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse


PROTECTED_PREFIXES = (
    "/chat",
    "/ask",
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
        return JSONResponse(
            {"detail": "Not found"},
            status_code=404,
        )

    if path == "/health":
        return JSONResponse({"status": "ok"})

    protected = any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in PROTECTED_PREFIXES
    )

    if protected:
        expected = os.getenv("GROWTHLAB_DEPLOY_TOKEN", "")

        if not expected:
            return JSONResponse(
                {"detail": "Service not configured"},
                status_code=503,
            )

        supplied = request.headers.get("Authorization", "")
        scheme, _, credential = supplied.partition(" ")

        if (
            scheme.lower() != "bearer"
            or not secrets.compare_digest(
                credential.strip(),
                expected,
            )
        ):
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

    return await call_next(request)
