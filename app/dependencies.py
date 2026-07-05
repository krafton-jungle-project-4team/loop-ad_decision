from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status

from app.config import Settings, load_settings


def get_settings(request: Request) -> Settings:
    settings = request.app.state.settings
    if settings is None:
        settings = load_settings()
        request.app.state.settings = settings
    return settings


def require_internal_key(
    request: Request,
    x_loop_ad_internal_key: str | None = Header(
        default=None,
        alias="X-Loop-Ad-Internal-Key",
    ),
) -> None:
    settings = get_settings(request)
    if not x_loop_ad_internal_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if not hmac.compare_digest(x_loop_ad_internal_key, settings.internal_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
