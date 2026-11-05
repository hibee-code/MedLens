"""The OpenAPI contract: committed document is current, and every route follows the conventions."""

import json
import re
from pathlib import Path
from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI

from medlens.core.config import Settings
from medlens.core.openapi import API_TAGS, problem_responses, render_openapi
from medlens.core.pagination import Page, PageParams, page_params
from medlens.core.routing import api_v1_router
from medlens.core.schemas import ReadModel
from medlens.main import create_app

OPENAPI_PATH = Path(__file__).parents[2] / "openapi.json"
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def operations(schema: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (method, path, operation)
        for path, item in schema["paths"].items()
        for method, operation in item.items()
        if method in HTTP_METHODS
    ]


def resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    return schema["components"]["schemas"][ref.rsplit("/", 1)[1]] if ref else node


@pytest.fixture
def schema(app: FastAPI) -> dict[str, Any]:
    return app.openapi()


def test_committed_openapi_is_current(settings_factory) -> None:
    current = render_openapi(create_app(settings_factory()))

    assert OPENAPI_PATH.read_text() == current, (
        "openapi.json is out of date. Run `make openapi`, review the diff and commit it."
    )


def test_metadata(schema: dict[str, Any]) -> None:
    assert schema["info"]["title"] == "MedLens API"
    assert "never a guarantee" in schema["info"]["description"]
    assert [tag["name"] for tag in schema["tags"]] == [
        *("auth", "users", "organizations", "products", "scans"),
        *("cases", "hotspots", "reports", "knowledge", "assistant", "admin", "health"),
    ]
    assert all(tag["description"] for tag in schema["tags"])
    assert schema["components"]["securitySchemes"]["bearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Short-lived access token from the auth endpoints.",
    }
    assert schema["components"]["securitySchemes"]["apiKeyAuth"]["name"] == "X-API-Key"


BINARY = {"type": "string", "format": "binary"}


def test_every_route_follows_the_conventions(schema: dict[str, Any]) -> None:
    known_tags = {tag["name"] for tag in API_TAGS}
    operation_ids = []
    for method, path, operation in operations(schema):
        where = f"{method.upper()} {path}"
        assert len(operation.get("tags", [])) == 1, f"{where}: needs exactly one tag"
        tag = operation["tags"][0]
        assert tag in known_tags, f"{where}: unknown tag {tag!r}"
        assert re.fullmatch(rf"{tag}-[a-z0-9_]+", operation["operationId"]), where
        assert operation.get("summary"), f"{where}: missing summary"
        assert operation.get("description"), f"{where}: missing description"

        success = [code for code in operation["responses"] if code.startswith("2")]
        assert success, f"{where}: no 2xx response"
        for code in success:
            if code == "204":
                assert not operation["responses"][code].get("content"), f"{where}: 204 has no body"
                continue
            content = operation["responses"][code].get("content", {})
            if "application/json" not in content and content:  # a download or an event stream
                for media_type, body in content.items():
                    expected = {"type": "string"} if media_type == "text/event-stream" else BINARY
                    assert body.get("schema") == expected, f"{where}: {media_type} schema"
                continue
            body = content.get("application/json", {})
            assert "schema" in body, f"{where}: {code} needs a response_model"
            documented = resolve(schema, body["schema"])
            if documented.get("type") == "array":  # a list: the item type carries the example
                documented = resolve(schema, documented["items"])
            page_items = documented.get("properties", {}).get("items", {})
            if page_items.get("type") == "array":  # a Page[T]: so does T
                documented = resolve(schema, page_items["items"])
            assert "examples" in documented or "example" in body or "examples" in body, (
                f"{where}: {code} response needs examples"
            )
        operation_ids.append(operation["operationId"])

    assert len(operation_ids) == len(set(operation_ids)), "operation ids must be unique"


def test_error_responses_are_problem_json(schema: dict[str, Any]) -> None:
    for method, path, operation in operations(schema):
        for code, response in operation["responses"].items():
            refs = json.dumps(response.get("content", {}))
            if "ProblemDetail" in refs:
                assert list(response["content"]) == ["application/problem+json"], (
                    f"{method.upper()} {path} {code}"
                )


def test_health_operation_ids(schema: dict[str, Any]) -> None:
    ids = {operation["operationId"] for _, _, operation in operations(schema)}

    assert {"health-liveness", "health-readiness"} <= ids


class Thing(ReadModel):
    name: str


@pytest.fixture
def sample_app(settings: Settings) -> FastAPI:
    app = create_app(settings)
    router = api_v1_router("/things", tags=["products"])

    @router.get(
        "",
        response_model=Page[Thing],
        summary="List things",
        description="Sample list route.",
        responses=problem_responses(401, 404),
    )
    async def list_things(params: Annotated[PageParams, Depends(page_params)]) -> Page[Thing]:
        raise NotImplementedError

    app.include_router(router)
    return app


def test_declared_errors_and_validation_use_problem_schemas(sample_app: FastAPI) -> None:
    schema = sample_app.openapi()
    responses = schema["paths"]["/api/v1/things"]["get"]["responses"]

    assert schema["paths"]["/api/v1/things"]["get"]["operationId"] == "products-list_things"
    for code, slug in (("401", "unauthorized"), ("404", "not-found"), ("422", "validation-failed")):
        body = responses[code]["content"]["application/problem+json"]
        assert body["example"]["type"] == f"urn:medlens:problem:{slug}"
    assert responses["422"]["content"]["application/problem+json"]["schema"] == {
        "$ref": "#/components/schemas/ValidationProblemDetail"
    }
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    assert "FieldError" in schema["components"]["schemas"]


def test_problem_responses_rejects_undocumented_status() -> None:
    with pytest.raises(ValueError, match="no problem documentation for status 418"):
        problem_responses(418)


def test_problem_type_prefix_is_written_once() -> None:
    src = Path(__file__).parents[2] / "src"
    files = [path for path in src.rglob("*.py") if "urn:medlens:problem" in path.read_text()]

    assert files == [src / "medlens" / "core" / "errors.py"]  # PROBLEM_TYPE_PREFIX
