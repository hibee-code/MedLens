"""OpenAPI document: metadata, conventions and post-processing.

Conventions every route follows (enforced by tests/unit/test_openapi.py):

* exactly one tag from ``API_TAGS``; the operation id is ``<tag>-<function name>``, so client
  generators get stable method names;
* a summary, a description, a ``response_model`` whose schema carries examples;
* documented errors via ``responses=problem_responses(401, 404, ...)``.

``customize_openapi`` then serves every error response as ``application/problem+json``, replaces
FastAPI's default 422 schema with ``ValidationProblemDetail``, and adds the security schemes.
The committed ``openapi.json`` is regenerated with ``make openapi``; a test fails when it drifts.
"""

import json
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute

from medlens.core.errors import (
    PROBLEM_CONTENT_TYPE,
    PROBLEM_TYPE_PREFIX,
    Conflict,
    DependencyUnavailable,
    DomainError,
    Forbidden,
    NotFound,
    PayloadTooLarge,
    ProblemDetail,
    RateLimited,
    Unauthorized,
    UnsupportedMediaType,
    ValidationFailed,
    ValidationProblemDetail,
)

API_TITLE = "MedLens API"

API_DESCRIPTION = f"""
MedLens checks a photo of a medicine pack against the register of NAFDAC-approved products and
returns a verdict with a confidence value:

* `LIKELY_GENUINE`: the pack matches a registered product and no risk signal fired.
* `SUSPICIOUS`: one or more risk signals fired, for example an expired registration, a batch
  seen in too many places, or a pack that does not look like the genuine reference.
* `UNREGISTERED`: no registered product matches the NAFDAC number on the pack.
* `INCONCLUSIVE`: the photo could not be read reliably; retake it in good light.

**A verdict is a risk assessment, never a guarantee that a medicine is genuine or safe to use.**
A `LIKELY_GENUINE` result only means no problem was detected. Suspected counterfeits should be
reported to NAFDAC; clients must present results with that caveat.

Conventions:

* Errors use [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) `application/problem+json` with
  a stable `type` URI (`{PROBLEM_TYPE_PREFIX}<slug>`) and a `request_id`.
* Send `X-Request-ID` (a UUID) to correlate requests; it is echoed on every response.
* Lists are cursor-paginated: pass `next_cursor` back as `cursor`.
""".strip()

API_TAGS: list[dict[str, str]] = [
    {"name": "auth", "description": "Phone OTP, staff login with MFA, tokens and API keys."},
    {"name": "users", "description": "The signed-in user's profile, data export and erasure."},
    {
        "name": "organizations",
        "description": "Pharmacies, manufacturers and regulators: members, invitations, branches.",
    },
    {
        "name": "products",
        "description": "Registered products by NAFDAC number, batches and reference pack images.",
    },
    {"name": "scans", "description": "Submit a pack photo and read the verification result."},
    {"name": "cases", "description": "Review queue for high-risk scans and lab outcomes."},
    {"name": "hotspots", "description": "Clusters of suspicious scans by place and time."},
    {
        "name": "reports",
        "description": "Aggregated statistics: the scan summary and manufacturers' reports.",
    },
    {
        "name": "knowledge",
        "description": "Regulatory documents for professional users: upload and hybrid search.",
    },
    {
        "name": "assistant",
        "description": "Answers grounded in the knowledge base, with citations (no scan verdicts).",
    },
    {"name": "admin", "description": "Platform administration: roles, permissions, operations."},
    {"name": "health", "description": "Liveness and readiness probes for orchestrators."},
]

SECURITY_SCHEMES: dict[str, dict[str, str]] = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Short-lived access token from the auth endpoints.",
    },
    "apiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Organization API key for machine clients: `ml_live_<prefix>_<secret>`.",
    },
}

# Status -> the DomainError raised for it (source of the documented type URI and title).
_DOCUMENTED_ERRORS: dict[int, type[DomainError]] = {
    error.status_code: error
    for error in (
        DomainError,
        Unauthorized,
        Forbidden,
        NotFound,
        Conflict,
        PayloadTooLarge,
        UnsupportedMediaType,
        ValidationFailed,
        RateLimited,
        DependencyUnavailable,
    )
}
_PROBLEM_DESCRIPTIONS = {
    400: "The request is malformed.",
    401: "Missing, expired or invalid credentials.",
    403: "Authenticated, but not allowed to do this.",
    404: "The resource does not exist or is not visible to the caller.",
    409: "The request conflicts with the current state of the resource.",
    413: "The request body or upload is too large.",
    415: "The upload is not an accepted file type.",
    422: "One or more fields are invalid; see `errors`.",
    429: "Rate limit exceeded; retry after the `Retry-After` header.",
    503: "A dependency is temporarily unavailable; retry later.",
}


def generate_operation_id(route: APIRoute) -> str:
    tag = str(route.tags[0]) if route.tags else "default"
    return f"{tag}-{route.name}"


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """Document error responses in one line: ``responses=problem_responses(401, 404)``."""
    responses: dict[int | str, dict[str, Any]] = {}
    for status in statuses:
        if status not in _DOCUMENTED_ERRORS:
            raise ValueError(f"no problem documentation for status {status}")
        model = (
            ValidationProblemDetail if status == HTTPStatus.UNPROCESSABLE_ENTITY else ProblemDetail
        )
        responses[status] = {
            "model": model,
            "description": _PROBLEM_DESCRIPTIONS[status],
            "content": {"application/json": {"example": _problem_example(status)}},
        }
    return responses


def _problem_example(status: int) -> dict[str, Any]:
    error = _DOCUMENTED_ERRORS[status]
    example: dict[str, Any] = {
        "type": f"{PROBLEM_TYPE_PREFIX}{error.slug}",
        "title": error.title,
        "status": status,
        "detail": _PROBLEM_DESCRIPTIONS[status],
        "instance": "/api/v1/scans/0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
        "request_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e13",
    }
    if status == HTTPStatus.UNPROCESSABLE_ENTITY:
        example["errors"] = [
            {
                "loc": ["query", "limit"],
                "msg": "Input should be less than or equal to 100",
                "type": "less_than_equal",
            }
        ]
    return example


def customize_openapi(schema: dict[str, Any]) -> dict[str, Any]:
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    components["securitySchemes"] = SECURITY_SCHEMES

    problem_refs = {
        f"#/components/schemas/{name}" for name in ("ProblemDetail", "ValidationProblemDetail")
    }
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            for status, response in operation.get("responses", {}).items():
                content = response.get("content", {})
                if status == "422" and _is_fastapi_default_422(content):
                    response["description"] = _PROBLEM_DESCRIPTIONS[422]
                    content["application/json"] = {
                        "schema": {"$ref": "#/components/schemas/ValidationProblemDetail"},
                        "example": _problem_example(422),
                    }
                json_body = content.get("application/json")
                if json_body and json_body.get("schema", {}).get("$ref") in problem_refs:
                    content[PROBLEM_CONTENT_TYPE] = content.pop("application/json")

    _add_model_schemas(schemas, ProblemDetail, ValidationProblemDetail)
    for unused in ("HTTPValidationError", "ValidationError"):
        if unused in schemas and f'"#/components/schemas/{unused}"' not in json.dumps(schema):
            del schemas[unused]
    return schema


def _is_fastapi_default_422(content: dict[str, Any]) -> bool:
    ref = content.get("application/json", {}).get("schema", {}).get("$ref", "")
    return bool(ref == "#/components/schemas/HTTPValidationError")


def _add_model_schemas(schemas: dict[str, Any], *models: type[ProblemDetail]) -> None:
    for model in models:
        model_schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        for name, definition in model_schema.pop("$defs", {}).items():
            schemas.setdefault(name, definition)
        schemas.setdefault(model.__name__, model_schema)


class MedLensAPI(FastAPI):
    """FastAPI with the customised OpenAPI document (built once, then cached)."""

    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema is None:
            self.openapi_schema = customize_openapi(super().openapi())
        return self.openapi_schema


def render_openapi(app: FastAPI) -> str:
    """The OpenAPI document exactly as committed to ``openapi.json``."""
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"
