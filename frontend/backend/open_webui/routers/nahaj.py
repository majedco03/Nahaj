"""Same-origin reverse proxy for the Nahaj domain API.

The browser only ever talks to Open WebUI.  This router injects the private
API bearer key server-side and forwards JSON, multipart uploads, downloads,
and streaming Chat Completions responses without exposing that key to the
client.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def _upstream_url(path: str, query: str) -> str:
    base = os.getenv("NAHAJ_API_URL", "http://api:8000").rstrip("/")
    url = f"{base}/{path.lstrip('/')}"
    return f"{url}?{query}" if query else url


def _forward_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in request.headers.items():
        if name.lower() not in {"host", "authorization", "content-length"}:
            headers[name] = value
    key = os.getenv("NAHAJ_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    """Forward a request to the private API, preserving streaming bodies."""

    client = httpx.AsyncClient(timeout=None, follow_redirects=False)
    upstream_request = client.build_request(
        request.method,
        _upstream_url(path, request.url.query),
        headers=_forward_headers(request),
        content=await request.body(),
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except Exception:
        await client.aclose()
        raise

    response_headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in _HOP_BY_HOP
    }

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=None,
    )
