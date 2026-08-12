from __future__ import annotations

import argparse
import json
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from yeetllm.state import DEFAULT_STATE_PATH, read_state

HOP_BY_HOP = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
BLOCKED_PATHS = {
    "/load_lora_adapter",
    "/start_profile",
    "/stop_profile",
    "/unload_lora_adapter",
    "/v1/load_lora_adapter",
    "/v1/unload_lora_adapter",
    "/wake_up",
    "/sleep",
}
RESPONSE_PATH = re.compile(r"^/v1/responses/([^/]+)(?:/cancel)?$")


class AffinityMap:
    def __init__(self, *, maximum: int = 10_000, ttl_seconds: float = 86_400) -> None:
        self.maximum = maximum
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def put(self, response_id: str, engine_id: str) -> None:
        self._expire()
        self._items[response_id] = (engine_id, time.monotonic() + self.ttl_seconds)
        self._items.move_to_end(response_id)
        while len(self._items) > self.maximum:
            self._items.popitem(last=False)

    def get(self, response_id: str) -> str | None:
        self._expire()
        item = self._items.get(response_id)
        if item is None:
            return None
        self._items.move_to_end(response_id)
        return item[0]

    def _expire(self) -> None:
        now = time.monotonic()
        expired = [key for key, (_, deadline) in self._items.items() if deadline <= now]
        for key in expired:
            self._items.pop(key, None)


class SSEIdParser:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)
        # SSE permits both LF and CRLF line endings. The parser observes a
        # private copy, so normalizing it does not alter the proxied bytes.
        self.buffer = bytearray(bytes(self.buffer).replace(b"\r\n", b"\n"))
        if len(self.buffer) > 128 * 1024:
            del self.buffer[: len(self.buffer) - 64 * 1024]
        while b"\n\n" in self.buffer:
            event, _, remaining = self.buffer.partition(b"\n\n")
            self.buffer = bytearray(remaining)
            for line in event.splitlines():
                if line.startswith(b"data:"):
                    self._parse_json(bytes(line[5:].strip()))

    def _parse_json(self, raw: bytes) -> None:
        if not raw or raw == b"[DONE]":
            return
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        response_id = find_response_id(payload)
        if response_id:
            self.callback(response_id)


class RouterContext:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.affinity = AffinityMap()
        timeout = httpx.Timeout(connect=10.0, pool=10.0, write=60.0, read=3600.0)
        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
        )

    async def close(self) -> None:
        await self.client.aclose()

    def state(self) -> dict[str, Any]:
        return read_state(self.state_path)


def create_app(state_path: Path = DEFAULT_STATE_PATH) -> Starlette:
    context = RouterContext(state_path)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        yield
        await context.close()

    async def live(_request: Request) -> Response:
        return JSONResponse({"status": "live"})

    async def ready(_request: Request) -> Response:
        try:
            state = context.state()
            is_ready, detail = readiness(state)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return JSONResponse({"status": "not_ready", "detail": str(exc)}, status_code=503)
        return JSONResponse(
            {"status": "ready" if is_ready else "not_ready", "detail": detail},
            status_code=200 if is_ready else 503,
        )

    async def models(_request: Request) -> Response:
        try:
            state = context.state()
            registry = state["registry"]
            created = int(registry["created"])
            records = registry["models"]
            cards = [
                {
                    "id": identifier,
                    "object": "model",
                    "created": created,
                    "owned_by": "yeetllm",
                    "root": identifier,
                    "parent": record.get("parent"),
                }
                for identifier, record in records.items()
            ]
            return JSONResponse({"object": "list", "data": cards})
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return openai_error("registry unavailable", "server_error", 503, detail=str(exc))

    async def proxy(request: Request) -> Response:
        if request.url.path.rstrip("/") in BLOCKED_PATHS:
            return openai_error(
                "This upstream management route is disabled by YeetLLM",
                "permission_error",
                403,
                code="management_route_disabled",
            )
        try:
            state = context.state()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return openai_error("registry unavailable", "server_error", 503, detail=str(exc))

        body = await request.body()
        requested_model = extract_model(request, body)
        affinity_engine: str | None = None
        if requested_model is None:
            match = RESPONSE_PATH.match(request.url.path)
            if match:
                affinity_engine = context.affinity.get(match.group(1))
                if affinity_engine is None:
                    return openai_error(
                        "The response ID is unknown to this router instance",
                        "invalid_request_error",
                        404,
                        code="response_not_found",
                    )

        registry = state.get("registry", {})
        model_records = registry.get("models", {})
        engine_records = registry.get("engines", {})
        if requested_model is not None:
            record = model_records.get(requested_model)
            if record is None:
                return openai_error(
                    f"The model {requested_model!r} does not exist",
                    "invalid_request_error",
                    404,
                    param="model",
                    code="model_not_found",
                )
            engine_id = record["engine_id"]
        elif affinity_engine is not None:
            engine_id = affinity_engine
        elif len(engine_records) == 1:
            engine_id = next(iter(engine_records))
        else:
            return openai_error(
                "A model is required when more than one engine is configured",
                "invalid_request_error",
                400,
                param="model",
                code="model_required",
            )

        runtime = state.get("engines", {}).get(engine_id, {})
        if runtime.get("state") != "ready":
            return openai_error(
                f"Engine for model {requested_model or engine_id!r} is unavailable",
                "server_error",
                503,
                code="engine_unavailable",
            )
        backend_url = engine_records[engine_id]["backend_url"]
        target = f"{backend_url}{request.url.path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        request_headers = filter_request_headers(request.headers.raw)
        upstream_request = context.client.build_request(
            request.method, target, headers=request_headers, content=body
        )
        try:
            upstream = await context.client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            return openai_error(
                "Could not connect to the selected inference engine",
                "server_error",
                503,
                code="upstream_unavailable",
                detail=str(exc),
            )

        response_headers = filter_response_headers(upstream.headers.raw)
        content_type = upstream.headers.get("content-type", "")
        if "text/event-stream" not in content_type.lower():
            try:
                if upstream.is_stream_consumed:
                    # In-memory/custom HTTPX transports may return an already-decoded
                    # response. Real network responses sent with stream=True take the
                    # raw branch below.
                    content = upstream.content
                    response_headers = [
                        item for item in response_headers if item[0].lower() != "content-encoding"
                    ]
                else:
                    content = b"".join([chunk async for chunk in upstream.aiter_raw()])
            finally:
                await upstream.aclose()
            remember_response_id(content, content_type, engine_id, context.affinity)
            return Response(
                content,
                status_code=upstream.status_code,
                headers=dict(response_headers),
            )

        parser = SSEIdParser(lambda response_id: context.affinity.put(response_id, engine_id))

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    if await request.is_disconnected():
                        break
                    parser.feed(chunk)
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream(), status_code=upstream.status_code, headers=dict(response_headers)
        )

    app = Starlette(
        routes=[
            Route("/health/live", live, methods=["GET"]),
            Route("/health/ready", ready, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route(
                "/v1/{path:path}",
                proxy,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            ),
        ],
        lifespan=lifespan,
    )
    app.state.yeetllm = context
    return app


def readiness(state: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    supervisor = state.get("supervisor", {})
    heartbeat = float(supervisor.get("heartbeat", 0))
    engines = state.get("engines", {})
    engine_states = {key: value.get("state", "unknown") for key, value in engines.items()}
    heartbeat_fresh = time.time() - heartbeat <= 10
    all_ready = bool(engines) and all(value == "ready" for value in engine_states.values())
    ready = heartbeat_fresh and supervisor.get("phase") == "ready" and all_ready
    return ready, {
        "phase": supervisor.get("phase", "unknown"),
        "heartbeat_fresh": heartbeat_fresh,
        "engines": engine_states,
    }


def extract_model(request: Request, body: bytes) -> str | None:
    query_model = request.query_params.get("model")
    if query_model:
        return query_model
    content_type = request.headers.get("content-type", "")
    lowered_content_type = content_type.lower()
    if "application/json" in lowered_content_type and body:
        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        model = data.get("model") if isinstance(data, dict) else None
        return model if isinstance(model, str) and model else None
    if "application/x-www-form-urlencoded" in lowered_content_type:
        values = parse_qs(body.decode("utf-8", errors="replace"))
        model_values = values.get("model")
        return model_values[0] if model_values else None
    if "multipart/form-data" in lowered_content_type:
        return extract_multipart_model(content_type, body)
    return None


def extract_multipart_model(content_type: str, body: bytes) -> str | None:
    match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type)
    if not match:
        return None
    boundary = (match.group(1) or match.group(2)).strip().encode()
    for part in body.split(b"--" + boundary):
        headers, separator, value = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        if re.search(br'content-disposition:[^\r\n]*name="model"', headers, re.IGNORECASE):
            if value.endswith(b"\r\n"):
                value = value[:-2]
            elif value.endswith(b"\n"):
                value = value[:-1]
            try:
                return value.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                return None
    return None


def filter_request_headers(raw: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    blocked = HOP_BY_HOP | connection_header_tokens(raw) | {b"host", b"content-length"}
    return [
        (name, value)
        for name, value in raw
        if name.lower() not in blocked
    ]


def filter_response_headers(raw: list[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
    blocked = HOP_BY_HOP | connection_header_tokens(raw) | {b"content-length"}
    return [
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in raw
        if name.lower() not in blocked
    ]


def connection_header_tokens(raw: list[tuple[bytes, bytes]]) -> set[bytes]:
    return {
        token.strip().lower()
        for name, value in raw
        if name.lower() == b"connection"
        for token in value.split(b",")
        if token.strip()
    }


def find_response_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    direct = payload.get("id")
    if isinstance(direct, str) and direct.startswith("resp_"):
        return direct
    response = payload.get("response")
    if isinstance(response, dict):
        nested = response.get("id")
        if isinstance(nested, str):
            return nested
    return None


def remember_response_id(
    content: bytes, content_type: str, engine_id: str, affinity: AffinityMap
) -> None:
    if "json" not in content_type.lower():
        return
    try:
        response_id = find_response_id(json.loads(content))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if response_id:
        affinity.put(response_id, engine_id)


def openai_error(
    message: str,
    error_type: str,
    status_code: int,
    *,
    param: str | None = None,
    code: str | None = None,
    detail: str | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }
    if detail:
        error["detail"] = detail
    return JSONResponse({"error": error}, status_code=status_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="YeetLLM internal router")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.state),
        host=args.host,
        port=args.port,
        access_log=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
