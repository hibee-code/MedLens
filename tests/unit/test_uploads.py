"""Upload validation contract: size, real type, polyglots, decompression bombs, re-encoding."""

import dataclasses
import hashlib
import io
import zipfile
from collections.abc import Iterator, Sequence
from typing import Annotated, Any

import httpx
import pytest
from fastapi import Depends, FastAPI, Response, UploadFile, params
from fastapi.routing import APIRoute
from PIL import ExifTags, Image
from starlette.datastructures import Headers
from structlog.testing import capture_logs

from medlens.core.config import Settings
from medlens.core.errors import ValidationFailed
from medlens.core.routing import api_v1_router
from medlens.core.uploads import (
    VALIDATED_UPLOADS,
    ValidatedImage,
    process_image,
    validate_image_upload,
)
from medlens.main import create_app

MAX_UPLOAD_BYTES = 200_000
MAX_PIXELS = 1_000_000


def make_image(fmt: str, size: tuple[int, int] = (64, 48), mode: str = "RGB", **save) -> bytes:
    color = (200, 30, 30, 128) if mode == "RGBA" else 128 if mode == "L" else (200, 30, 30)
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, fmt, **save)
    return buffer.getvalue()


def jpeg_with_gps_and_rotation() -> bytes:
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6  # stored sideways: rotate 90 degrees to display
    exif[ExifTags.Base.Make] = "PhoneMaker"
    exif.get_ifd(ExifTags.IFD.GPSInfo).update(
        {
            ExifTags.GPS.GPSLatitudeRef: "N",
            ExifTags.GPS.GPSLatitude: (6.0, 31.0, 28.0),
            ExifTags.GPS.GPSLongitudeRef: "E",
            ExifTags.GPS.GPSLongitude: (3.0, 22.0, 45.0),
        }
    )
    return make_image("JPEG", size=(64, 48), exif=exif)


@pytest.fixture
def settings(settings_factory) -> Settings:
    return settings_factory(
        app={
            "max_request_body_bytes": 1_000_000,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_image_pixels": MAX_PIXELS,
        }
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    app = create_app(settings)
    router = api_v1_router("/_test", tags=["scans"])

    @router.post("/upload")
    async def upload(image: Annotated[ValidatedImage, Depends(validate_image_upload)]) -> Response:
        headers = {
            "X-Width": str(image.width),
            "X-Height": str(image.height),
            "X-Sha256": image.sha256,
            "X-Source-Type": image.source_type,
        }
        return Response(image.jpeg, media_type="image/jpeg", headers=headers)

    app.include_router(router)
    return app


async def upload(
    client: httpx.AsyncClient,
    data: bytes,
    filename: str = "pack.jpg",
    content_type: str = "image/jpeg",
) -> httpx.Response:
    files = {"image": (filename, data, content_type)}
    return await client.post("/api/v1/_test/upload", files=files)


# ------------------------------------------------------------------------- accepted
@pytest.mark.parametrize(
    ("fmt", "filename", "content_type"),
    [
        ("JPEG", "pack.jpg", "image/jpeg"),
        ("PNG", "pack.png", "image/png"),
        ("WEBP", "pack.webp", "image/webp"),
        ("JPEG", "blob", "application/octet-stream"),  # nothing declared: sniffing decides
        ("PNG", "IMG_0001", "image/png"),
    ],
)
async def test_accepted_images_are_reencoded_as_jpeg(
    client: httpx.AsyncClient, fmt: str, filename: str, content_type: str
) -> None:
    response = await upload(client, make_image(fmt), filename, content_type)

    assert response.status_code == 200, response.text
    output = Image.open(io.BytesIO(response.content))
    assert output.format == "JPEG"
    assert output.mode == "RGB"
    assert output.size == (64, 48)
    assert response.headers["x-sha256"] == hashlib.sha256(response.content).hexdigest()
    assert response.headers["x-source-type"] == f"image/{fmt.lower()}"


async def test_exif_and_gps_are_stripped_after_applying_orientation(
    client: httpx.AsyncClient,
) -> None:
    original = jpeg_with_gps_and_rotation()
    assert Image.open(io.BytesIO(original)).getexif().get_ifd(ExifTags.IFD.GPSInfo)  # sanity

    response = await upload(client, original)

    assert response.status_code == 200
    output = Image.open(io.BytesIO(response.content))
    assert output.size == (48, 64)  # rotated upright before the orientation tag was dropped
    assert dict(output.getexif()) == {}
    assert b"Exif" not in response.content
    assert b"PhoneMaker" not in response.content


async def test_transparency_is_flattened_onto_white(client: httpx.AsyncClient) -> None:
    transparent = make_image("PNG", mode="RGBA")

    response = await upload(client, transparent, "pack.png", "image/png")

    pixel = Image.open(io.BytesIO(response.content)).getpixel((10, 10))
    assert pixel == pytest.approx((228, 143, 143), abs=3)  # 50% red over white


# ------------------------------------------------------------------------- wrong type
async def test_png_renamed_to_jpg_is_rejected(client: httpx.AsyncClient) -> None:
    response = await upload(client, make_image("PNG"), "pack.jpg", "image/jpeg")

    assert response.status_code == 415
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:medlens:problem:unsupported-media-type"


async def test_extension_mismatch_alone_is_rejected(client: httpx.AsyncClient) -> None:
    response = await upload(client, make_image("PNG"), "pack.jpg", "application/octet-stream")

    assert response.status_code == 415


@pytest.mark.parametrize(
    ("data", "filename", "content_type"),
    [
        (b"MZ\x90\x00 definitely not an image" * 10, "pack.jpg", "image/jpeg"),
        (b"<html><body>hi</body></html>", "pack.jpg", "image/jpeg"),
        (make_image("GIF"), "pack.gif", "image/gif"),
        (make_image("BMP"), "pack.bmp", "image/bmp"),
    ],
)
async def test_non_image_or_disallowed_types_are_rejected(
    client: httpx.AsyncClient, data: bytes, filename: str, content_type: str
) -> None:
    response = await upload(client, data, filename, content_type)

    assert response.status_code == 415


# ------------------------------------------------------------------------- polyglots
def _zip_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("payload.sh", "#!/bin/sh\necho owned\n")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("data", "filename", "content_type"),
    [
        (make_image("JPEG") + _zip_archive(), "pack.jpg", "image/jpeg"),
        (make_image("JPEG", comment=b"<script>alert(1)</script>"), "pack.jpg", "image/jpeg"),
        (make_image("JPEG", comment=b"%PDF-1.7 hidden document"), "pack.jpg", "image/jpeg"),
        (make_image("JPEG") + b"<?php system($_GET['c']); ?>", "pack.jpg", "image/jpeg"),
        (make_image("PNG") + b"\x00" * 32, "pack.png", "image/png"),
        (make_image("PNG") + b"<html><body>x</body></html>", "pack.png", "image/png"),
        (make_image("WEBP") + b"trailing data", "pack.webp", "image/webp"),
    ],
    ids=[
        "jpeg+zip",
        "jpeg+script",
        "jpeg+pdf",
        "jpeg+php",
        "png+trailing",
        "png+html",
        "webp+trailing",
    ],
)
async def test_polyglot_files_are_rejected(
    client: httpx.AsyncClient, data: bytes, filename: str, content_type: str
) -> None:
    with capture_logs() as logs:
        response = await upload(client, data, filename, content_type)

    assert response.status_code == 422
    assert response.json()["errors"][0]["loc"] == ["body", "image"]
    assert [log["reason"] for log in logs if log["event"] == "upload.rejected"]


async def test_jpeg_with_trailing_camera_data_is_accepted(client: httpx.AsyncClient) -> None:
    # Android phones append depth maps / motion-photo video after the JPEG end marker.
    motion_photo = make_image("JPEG") + b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 4

    response = await upload(client, motion_photo)

    assert response.status_code == 200
    assert b"ftypmp42" not in response.content  # only re-encoded pixels come out


# ------------------------------------------------------------------------- size and bombs
async def test_oversized_upload_is_rejected(client: httpx.AsyncClient) -> None:
    big = make_image("PNG") + b"\x00" * MAX_UPLOAD_BYTES

    response = await upload(client, big, "pack.png", "image/png")

    assert response.status_code == 413
    assert response.json()["type"] == "urn:medlens:problem:payload-too-large"


async def test_empty_upload_is_rejected(client: httpx.AsyncClient) -> None:
    assert (await upload(client, b"")).status_code == 422


@pytest.mark.parametrize(
    "size",
    [
        (1100, 1000),  # just over the limit: caught by the explicit dimension check
        (3000, 3000),  # far over: Pillow's DecompressionBombError
    ],
)
@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
async def test_decompression_bomb_is_rejected(
    client: httpx.AsyncClient, size: tuple[int, int]
) -> None:
    bomb = make_image("PNG", size=size, mode="L", optimize=True)
    assert len(bomb) < MAX_UPLOAD_BYTES  # tiny file, huge image

    response = await upload(client, bomb, "pack.png", "image/png")

    assert response.status_code == 422
    assert response.json()["detail"] == "The image has too many pixels."


async def test_image_at_the_pixel_limit_is_accepted(client: httpx.AsyncClient) -> None:
    response = await upload(
        client, make_image("PNG", size=(1000, 1000), mode="L"), "a.png", "image/png"
    )

    assert response.status_code == 200


async def test_truncated_image_is_rejected(client: httpx.AsyncClient) -> None:
    whole = make_image("JPEG", size=(400, 300), quality=95)

    response = await upload(client, whole[: len(whole) // 2])

    assert response.status_code == 422
    assert response.json()["detail"] == "The image could not be decoded."


def test_animated_images_are_rejected() -> None:
    frames = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
    buffer = io.BytesIO()
    frames[0].save(buffer, "PNG", save_all=True, append_images=frames[1:])

    with pytest.raises(ValidationFailed, match="Animated images are not accepted"):
        process_image(
            buffer.getvalue(), declared_type="image/png", filename=None, max_pixels=MAX_PIXELS
        )


# ------------------------------------------------------------------------- only re-encoded bytes
def test_validated_image_has_no_field_for_the_original() -> None:
    fields = {field.name for field in dataclasses.fields(ValidatedImage)}

    assert fields == {"jpeg", "width", "height", "sha256", "source_type"}


async def test_validator_closes_the_original_upload(settings: Settings) -> None:
    original = make_image("PNG")
    upload_file = UploadFile(
        io.BytesIO(original), filename="a.png", headers=Headers({"content-type": "image/png"})
    )

    result = await validate_image_upload(upload_file, settings)

    assert upload_file.file.closed
    assert result.jpeg != original
    assert result.jpeg.startswith(b"\xff\xd8")  # JPEG


def _api_routes(routes: Sequence[Any]) -> Iterator[APIRoute]:
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        elif (router := getattr(route, "original_router", None)) is not None:
            yield from _api_routes(router.routes)  # FastAPI keeps included routers unflattened


def _file_params_outside_validator(dependant: Any) -> list[str]:
    if dependant.call in VALIDATED_UPLOADS:
        return []
    found = [f.name for f in dependant.body_params if isinstance(f.field_info, params.File)]
    for sub_dependant in dependant.dependencies:
        found += _file_params_outside_validator(sub_dependant)
    return found


def upload_violations(app: FastAPI) -> list[str]:
    """Routes that could see an original upload instead of the re-encoded image."""
    return [
        f"{route.path}: {names}"
        for route in _api_routes(app.routes)
        if (names := _file_params_outside_validator(route.dependant))
    ]


def test_no_route_accepts_files_except_through_the_validator(settings: Settings) -> None:
    assert upload_violations(create_app(settings)) == []


def test_upload_guard_detects_raw_file_parameters(app: FastAPI) -> None:
    router = api_v1_router("/_raw", tags=["scans"])

    @router.post("/upload")
    async def raw_upload(image: UploadFile) -> dict[str, int]:
        return {"size": len(await image.read())}

    app.include_router(router)

    assert upload_violations(app) == ["/api/v1/_raw/upload: ['image']"]


async def test_body_limit_applies_before_upload_parsing(client: httpx.AsyncClient) -> None:
    oversized_body = b"\x00" * 1_000_001  # over max_request_body_bytes, not just the upload limit

    with capture_logs() as logs:
        response = await upload(client, oversized_body, "pack.png", "image/png")

    assert response.status_code == 413
    assert response.json()["detail"] == "The request body exceeds the limit of 1000000 bytes."
    assert not [log for log in logs if log["event"] == "upload.rejected"]  # never parsed
