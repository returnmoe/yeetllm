from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

from yeetllm.config import YeetConfig
from yeetllm.registry import build_registry
from yeetllm.router import create_app
from yeetllm.state import StateStore


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def write_ready_state(path: Path, *, engine_state: str = "ready") -> None:
    config = YeetConfig.model_validate(
        {
            "models": [
                {
                    "id": "qwen",
                    "model": "org/qwen",
                    "gpus": [0],
                    "lora": {
                        "enabled": True,
                        "max_loras": 1,
                        "max_cpu_loras": 1,
                        "adapters": [{"id": "qwen-code", "model": "org/lora"}],
                    },
                }
            ]
        }
    )
    store = StateStore(path)
    store.initialize(
        {
            "supervisor": {"phase": "ready", "heartbeat": time.time()},
            "registry": build_registry(config).as_state(),
            "engines": {"qwen": {"state": engine_state}},
        }
    )


async def router_client(
    tmp_path: Path, handler: Any, *, engine_state: str = "ready"
) -> tuple[httpx.AsyncClient, Any]:
    state_path = tmp_path / "state.json"
    write_ready_state(state_path, engine_state=engine_state)
    app = create_app(state_path)
    await app.state.yeetllm.client.aclose()
    app.state.yeetllm.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://upstream"
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    )
    return client, app


@pytest.mark.asyncio
async def test_non_streaming_proxy_preserves_body_status_and_headers(tmp_path: Path) -> None:
    observed: dict[str, Any] = {}

    async def upstream(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["body"] = await request.aread()
        observed["headers"] = dict(request.headers)
        return httpx.Response(
            429,
            content=b'{"error":{"message":"busy"}}',
            headers={
                "connection": "x-upstream-private",
                "content-type": "application/json",
                "x-request-id": "upstream-1",
                "x-upstream-private": "remove-me",
            },
        )

    client, app = await router_client(tmp_path, upstream)
    try:
        body = {"model": "qwen-code", "messages": [{"role": "user", "content": "hi"}]}
        response = await client.post(
            "/v1/chat/completions?x=1",
            json=body,
            headers={"connection": "x-client-private", "x-client-private": "remove-me"},
        )
        assert response.status_code == 429
        assert response.headers["x-request-id"] == "upstream-1"
        assert "x-upstream-private" not in response.headers
        assert response.json() == {"error": {"message": "busy"}}
        assert observed["url"] == "http://127.0.0.1:8100/v1/chat/completions?x=1"
        assert json.loads(observed["body"])["model"] == "qwen-code"
        assert "x-client-private" not in observed["headers"]
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_non_streaming_encoded_body_is_not_double_decoded(tmp_path: Path) -> None:
    plain = b'{"id":"chatcmpl_123","choices":[]}'

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=gzip.compress(plain),
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )

    client, app = await router_client(tmp_path, upstream)
    try:
        response = await client.post("/v1/chat/completions", json={"model": "qwen"})
        assert response.status_code == 200
        assert response.content == plain
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_multipart_model_routing_preserves_body_and_case_sensitive_boundary(
    tmp_path: Path,
) -> None:
    boundary = "YeetBoundaryABC"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        "qwen-code-\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    async def upstream(request: httpx.Request) -> httpx.Response:
        assert await request.aread() == body
        return httpx.Response(200, json={"text": "ok"})

    client, app = await router_client(tmp_path, upstream)
    try:
        state_path = tmp_path / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        lora = state["registry"]["models"].pop("qwen-code")
        lora["id"] = "qwen-code-"
        state["registry"]["models"]["qwen-code-"] = lora
        state_path.write_text(json.dumps(state), encoding="utf-8")

        response = await client.post(
            "/v1/audio/transcriptions",
            content=body,
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
        assert response.status_code == 200
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_sse_stream_is_proxied_and_closed(tmp_path: Path) -> None:
    stream = Chunks(
        [
            b'data: {"type":"response.created","response":{"id":"resp_123"}}\r\n',
            b'\r\ndata: {"type":"response.output_text.delta","delta":"hi"}\r\n\r\n',
            b"data: [DONE]\r\n\r\n",
        ]
    )

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    client, app = await router_client(tmp_path, upstream)
    try:
        async with client.stream("POST", "/v1/responses", json={"model": "qwen"}) as response:
            chunks = [chunk async for chunk in response.aiter_bytes()]
        assert response.status_code == 200
        assert b"".join(chunks).endswith(b"data: [DONE]\r\n\r\n")
        assert stream.closed

        async def continuation(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/responses/resp_123/cancel"
            return httpx.Response(200, json={"id": "resp_123", "status": "cancelled"})

        await app.state.yeetllm.client.aclose()
        app.state.yeetllm.client = httpx.AsyncClient(
            transport=httpx.MockTransport(continuation)
        )
        cancel = await client.post("/v1/responses/resp_123/cancel")
        assert cancel.status_code == 200
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_client_disconnect_closes_streaming_upstream(tmp_path: Path) -> None:
    stream = Chunks([b'data: {"id":"chatcmpl_123"}\n\n', b"data: [DONE]\n\n"])

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    client, app = await router_client(tmp_path, upstream)
    messages: list[dict[str, Any]] = [
        {
            "type": "http.request",
            "body": b'{"model":"qwen"}',
            "more_body": False,
        },
        {"type": "http.disconnect"},
    ]

    async def receive() -> dict[str, Any]:
        return messages.pop(0)

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("router", 80),
        },
        receive,
    )
    route = next(route for route in app.routes if getattr(route, "path", None) == "/v1/{path:path}")
    try:
        response = await route.endpoint(request)
        assert isinstance(response, StreamingResponse)
        chunks = [chunk async for chunk in response.body_iterator]
        assert chunks == []
        assert stream.closed
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_upstream_connection_failure_is_openai_error(tmp_path: Path) -> None:
    async def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client, app = await router_client(tmp_path, upstream)
    try:
        response = await client.post("/v1/completions", json={"model": "qwen"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "upstream_unavailable"
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_unknown_model_and_unavailable_engine_are_openai_errors(tmp_path: Path) -> None:
    async def upstream(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be contacted")

    client, app = await router_client(tmp_path, upstream)
    try:
        unknown = await client.post("/v1/completions", json={"model": "missing"})
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "model_not_found"
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()

    client, app = await router_client(tmp_path, upstream, engine_state="failed")
    try:
        unavailable = await client.post("/v1/embeddings", json={"model": "qwen"})
        assert unavailable.status_code == 503
        assert unavailable.json()["error"]["code"] == "engine_unavailable"
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_models_aggregates_base_and_lora(tmp_path: Path) -> None:
    async def upstream(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("/v1/models is local")

    client, app = await router_client(tmp_path, upstream)
    try:
        response = await client.get("/v1/models")
        assert response.status_code == 200
        assert [item["id"] for item in response.json()["data"]] == ["qwen", "qwen-code"]
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_liveness_is_local_and_readiness_requires_all_engines(tmp_path: Path) -> None:
    async def upstream(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("health routes are local")

    client, app = await router_client(tmp_path, upstream)
    try:
        live = await client.get("/health/live")
        ready = await client.get("/health/ready")
        assert live.status_code == 200
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"

        write_ready_state(tmp_path / "state.json", engine_state="failed")
        degraded = await client.get("/health/ready")
        assert degraded.status_code == 503
        assert degraded.json()["detail"]["engines"] == {"qwen": "failed"}
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()


@pytest.mark.asyncio
async def test_management_routes_are_not_exposed(tmp_path: Path) -> None:
    async def upstream(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("blocked route must not reach upstream")

    client, app = await router_client(tmp_path, upstream)
    try:
        response = await client.post("/v1/load_lora_adapter", json={"model": "qwen"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "management_route_disabled"
        trailing = await client.post(
            "/v1/load_lora_adapter/", json={"model": "qwen"}
        )
        assert trailing.status_code == 403
    finally:
        await client.aclose()
        await app.state.yeetllm.client.aclose()
