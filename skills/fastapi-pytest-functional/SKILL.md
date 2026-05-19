---
name: fastapi-pytest-functional
description: Function-style (NOT class-based) pytest for FastAPI apps and general HTTP API testing. Covers TestClient vs AsyncClient/httpx, fixtures in conftest.py, dependency_overrides, parametrize for matrix tests, async tests with anyio/asyncio, mocking external services with respx/pytest-mock, JSON-schema/Pydantic response validation, SSE/WebSocket testing, and CI-friendly setup. Use when writing or reviewing pytest suites for FastAPI services or any HTTP API. NOT for Django.
---

# FastAPI + pytest — Functional Style

Every test is a **plain function** named `test_…`. No `class TestX:`
wrappers, no `unittest.TestCase`. Fixtures handle setup; functions
stay short and read like documentation of behaviour.

## Why functional, not class-based

- **Less indirection.** Function name + fixture names tell you everything; no need to scroll for `setUp`.
- **Fixtures compose.** Class hierarchies don't. Multiple fixtures per test is normal and clean.
- **Better parametrize ergonomics.** `@pytest.mark.parametrize` on a function is one line; on a method it has to dodge `self`.
- **Easier to grep.** `def test_returns_404_when_user_missing` is one searchable line.

If you find yourself writing a `class TestUsers` to "share state",
that state is a fixture in disguise — extract it.

---

## Project layout

```
tests/
├── conftest.py             # shared fixtures (app, client, db)
├── unit/                   # pure functions, no I/O
│   └── test_pricing.py
├── api/                    # in-process via TestClient
│   ├── test_health.py
│   ├── test_users.py
│   └── test_orders.py
└── integration/            # talks to real Postgres/Redis (docker-compose)
    └── test_user_flow.py
```

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"            # @pytest.mark.asyncio not needed
testpaths    = ["tests"]
addopts      = "-q --strict-markers"
markers      = [
  "slow: marks tests as slow (deselect with '-m \"not slow\"')",
  "integration: requires docker services",
]
```

---

## Core fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app           # your FastAPI() instance
from app.deps import get_db        # whatever you DI


# ---- sync client (most tests use this) ---------------------------------
@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


# ---- async client (for streaming / SSE / WebSocket / native async tests)
@pytest.fixture
async def aclient() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---- per-test fake DB via dependency_overrides --------------------------
@pytest.fixture
def fake_db():
    return {"users": {}, "orders": {}}


@pytest.fixture
def app_with_fake_db(fake_db):
    def _override():
        return fake_db
    app.dependency_overrides[get_db] = _override
    yield app
    app.dependency_overrides.clear()
```

`TestClient` wraps `httpx` synchronously — perfect for 90% of API
tests. Switch to `AsyncClient` only when you actually need async
(SSE streaming, WebSockets, or you call `await` on something in
the test itself).

---

## Anatomy of a clean test

```python
# tests/api/test_users.py

def test_create_user_returns_201(client, app_with_fake_db):
    resp = client.post("/users", json={"email": "a@b.com", "name": "Aliya"})

    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "a@b.com"
    assert "id" in body


def test_create_user_rejects_invalid_email(client):
    resp = client.post("/users", json={"email": "not-an-email", "name": "x"})
    assert resp.status_code == 422
    # FastAPI's pydantic-driven error shape
    assert resp.json()["detail"][0]["loc"] == ["body", "email"]


def test_get_user_404_when_missing(client, app_with_fake_db):
    assert client.get("/users/does-not-exist").status_code == 404
```

Three things every test name should answer: **subject** (what
endpoint/feature), **condition** (what you fed it), **outcome**
(what you expect).

`test_<subject>_<outcome>_when_<condition>` — pick a pattern and
keep it across the suite.

---

## Parametrize — the highest-leverage feature

```python
import pytest

@pytest.mark.parametrize(
    "email,status",
    [
        ("a@b.com",      201),
        ("a+tag@b.com",  201),
        ("no-at-sign",   422),
        ("",             422),
        ("@b.com",       422),
    ],
)
def test_email_validation(client, email, status):
    resp = client.post("/users", json={"email": email, "name": "x"})
    assert resp.status_code == status
```

For named cases (clearer failure output):

```python
@pytest.mark.parametrize(
    "payload,expected_status",
    [
        pytest.param({"email": "a@b.com"},      201, id="valid"),
        pytest.param({"email": ""},             422, id="empty-email"),
        pytest.param({"email": "no-at"},        422, id="malformed-email"),
        pytest.param({},                         422, id="missing-email"),
    ],
)
def test_create_user(client, payload, expected_status):
    assert client.post("/users", json=payload).status_code == expected_status
```

---

## Async tests (when you really need them)

```python
# pyproject sets asyncio_mode = "auto" — no decorator needed
async def test_stream_chunks(aclient):
    chunks = []
    async with aclient.stream("POST", "/stream", json={"prompt": "hi"}) as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line.removeprefix("data: "))
    assert len(chunks) > 0
    assert chunks[-1] == "[DONE]"
```

For SSE specifically, look for `data:` lines and an explicit end
marker (e.g. `[DONE]`) in your app's response shape.

---

## Mocking external HTTP — `respx`

```python
import respx, httpx

def test_calls_billing_api(client):
    with respx.mock(base_url="https://billing.example") as mock:
        mock.post("/charge").mock(return_value=httpx.Response(200, json={"id": "ch_1"}))

        resp = client.post("/orders", json={"amount": 50, "user_id": "u1"})

        assert resp.status_code == 201
        assert mock["/charge"].called
        # body sent to billing API:
        sent = mock["/charge"].calls.last.request.read()
        assert b'"amount":50' in sent
```

`respx` intercepts all `httpx` calls — fast, deterministic, no real
network. For `requests`-based code, use `responses` instead.

---

## Mocking application code — `pytest-mock` / `monkeypatch`

```python
def test_uses_cached_result(client, monkeypatch):
    calls = []

    def fake_embed(text):
        calls.append(text)
        return [0.1] * 384

    monkeypatch.setattr("app.services.embeddings.embed", fake_embed)

    client.post("/embed", json={"text": "hello"})
    client.post("/embed", json={"text": "hello"})

    assert len(calls) == 1, "second call should hit cache"
```

`monkeypatch` is stdlib-shaped (built into pytest). `mocker` from
`pytest-mock` is the same with nicer ergonomics
(`mocker.patch("app.x.y", return_value=...)`).

---

## Schema / contract assertions

Stop checking individual fields one by one — round-trip through the
Pydantic schema:

```python
from app.schemas import UserResponse

def test_create_user_response_shape(client, app_with_fake_db):
    body = client.post("/users", json={"email": "a@b.com", "name": "Aliya"}).json()
    # raises if shape drifts from the schema
    user = UserResponse.model_validate(body)
    assert user.email == "a@b.com"
```

For external APIs you don't control, use `jsonschema` with a
schema file in `tests/schemas/`.

---

## Database tests — real DB or in-memory

**Option A (recommended): real Postgres in CI.** Spin it up with
`docker-compose` or `testcontainers-python`; commit a `tests/sql/seed.sql`.

```python
@pytest.fixture(scope="session")
def pg_url():
    return os.environ["TEST_POSTGRES_URL"]

@pytest.fixture
def db(pg_url):
    eng = create_engine(pg_url)
    with eng.connect() as conn:
        trans = conn.begin()
        yield conn
        trans.rollback()      # nothing persists between tests
```

**Option B: SQLAlchemy with SQLite-in-memory.** Fast, but
semantics drift from Postgres (UUIDs, JSONB, `RETURNING`). Use
only for unit-level tests of pure SQLA code.

---

## Auth / headers — use a fixture

```python
@pytest.fixture
def auth_headers() -> dict:
    return {"Authorization": "Bearer test-token"}

def test_protected(client, auth_headers):
    assert client.get("/me", headers=auth_headers).status_code == 200
```

For more complex flows, build a `logged_in_client` fixture that
returns a `TestClient` with the header baked in:

```python
@pytest.fixture
def logged_in_client(client, auth_headers):
    client.headers.update(auth_headers)
    return client
```

---

## WebSocket tests

```python
def test_chat_websocket(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        msg = ws.receive_json()
        assert msg == {"type": "pong"}
```

`TestClient.websocket_connect` is a context manager — close happens
automatically.

---

## Coverage and selection

```bash
uv run pytest                              # everything
uv run pytest tests/api/                   # one folder
uv run pytest -k "create_user"             # name match
uv run pytest -m "not slow"                # skip slow marker
uv run pytest --cov=app --cov-report=html  # coverage
uv run pytest -x --pdb                     # stop at first failure, drop into pdb
```

Aim for `≥ 85%` line coverage on application code. Don't chase 100%
— some glue isn't worth it.

---

## Anti-patterns

| Anti-pattern | Fix |
|---|---|
| `class TestUsers:` with `setUp` | Use fixtures + plain functions |
| Real network calls in `tests/api/` | `respx.mock` |
| Time-based tests using `time.sleep` | Freeze time with `freezegun` or `time-machine` |
| `assert response.status_code == 200` for *every* call | Add a `created()` helper that asserts 201 + returns body |
| Shared mutable state between tests | Use fixtures with proper scope (`function` by default) |
| Long tests with multiple acts | Split — one act per test, named for it |
| Mocks that mock too much (mock chain 4 deep) | The code is too coupled. Refactor it. |

---

## A complete real test file

```python
# tests/api/test_orders.py
import pytest, respx, httpx
from app.schemas import OrderResponse


def test_create_order_returns_201(client, app_with_fake_db):
    with respx.mock(base_url="https://billing.example") as bills:
        bills.post("/charge").mock(return_value=httpx.Response(200, json={"id": "ch_1"}))

        resp = client.post("/orders", json={"amount": 99.0, "user_id": "u1"})

    assert resp.status_code == 201
    order = OrderResponse.model_validate(resp.json())
    assert order.charge_id == "ch_1"


@pytest.mark.parametrize(
    "amount,status",
    [
        pytest.param(0.0,     422, id="zero"),
        pytest.param(-1.0,    422, id="negative"),
        pytest.param(10_001,  422, id="over-cap"),
    ],
)
def test_create_order_rejects_bad_amounts(client, amount, status):
    assert client.post("/orders", json={"amount": amount, "user_id": "u1"}).status_code == status


def test_get_order_404(client, app_with_fake_db):
    assert client.get("/orders/missing").status_code == 404
```

That's the entire shape: plain functions, fixtures injected by name,
parametrize for matrices, respx for outbound HTTP, Pydantic for
response shape. No `class`, no `unittest`.
