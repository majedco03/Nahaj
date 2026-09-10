"""Boundary contract for the separately-owned Nahaj supervisor.

This module deliberately contains no agent implementation.  The API imports
only these transport types and asks an injected gateway to handle requests.
Set ``SUPERVISOR_FACTORY=module.path:factory`` in the API environment when the
agentic system is ready to provide that gateway.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4


RequestType = Literal["chat", "document_ingest", "document_remove", "progress_check"]


@dataclass(slots=True)
class SupervisorRequest:
    type: RequestType
    request_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    document: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None


@dataclass(slots=True)
class SupervisorEvent:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class SupervisorGateway(Protocol):
    async def handle(self, request: SupervisorRequest) -> AsyncIterator[SupervisorEvent]:
        """Yield supervisor events for one API request."""


class SupervisorUnavailable(RuntimeError):
    """Raised when the agentic system has not been connected yet."""


_gateway: SupervisorGateway | None = None
_gateway_path = ""


def get_supervisor() -> SupervisorGateway:
    global _gateway, _gateway_path
    factory_path = os.getenv("SUPERVISOR_FACTORY", "").strip()
    if not factory_path:
        _gateway = None
        _gateway_path = ""
        raise SupervisorUnavailable(
            "The supervisor gateway is not configured. Set SUPERVISOR_FACTORY to the "
            "agentic system's gateway factory."
        )
    if _gateway is not None and _gateway_path == factory_path:
        return _gateway
    module_name, separator, attr_name = factory_path.partition(":")
    if not separator or not module_name or not attr_name:
        raise SupervisorUnavailable("SUPERVISOR_FACTORY must use module.path:factory_name")
    try:
        factory = getattr(importlib.import_module(module_name), attr_name)
        gateway = factory() if callable(factory) else factory
    except Exception as exc:  # Keep API errors stable at the boundary.
        raise SupervisorUnavailable("The supervisor gateway could not be loaded.") from exc
    if not hasattr(gateway, "handle"):
        raise SupervisorUnavailable("The supervisor gateway must expose an async handle method.")
    _gateway = gateway
    _gateway_path = factory_path
    return gateway
