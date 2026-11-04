"""Image upload validation.

The API accepts a pack photo only after it passes, in order:

1. size     - at most ``MEDLENS_APP__MAX_UPLOAD_BYTES`` (413);
2. type     - the real type, sniffed from the first bytes with libmagic, is JPEG, PNG or WebP,
              and agrees with any type the client declared (415);
3. polyglot - no embedded markup/script, PDF header, ZIP archive, or data appended after a PNG
              or WebP image (422). JPEGs may carry trailing data: Android phones append depth
              maps and motion-photo video after the image, so only the signature checks apply;
4. decode   - Pillow decodes it fully, refusing images over ``MEDLENS_APP__MAX_IMAGE_PIXELS``
              before decoding (decompression bombs) and animated images (422);
5. re-encode - EXIF orientation is applied, then the image is re-encoded as a baseline JPEG
              with no metadata at all (no EXIF, so no GPS), and hashed with SHA-256.

Only the re-encoded bytes are ever stored or passed on, never the original upload:
``ValidatedImage`` holds no reference to the original, ``validate_image_upload`` closes the upload
as soon as it has read it, and tests/unit/test_uploads.py fails if any route accepts a file
other than through ``VALIDATED_UPLOADS`` (this, ``read_csv_upload`` for CSV data files and
``read_document_upload`` for knowledge-base documents).
The polyglot checks are defence in depth on top of re-encoding, which already discards
anything that is not pixels.

Decoding is CPU work, so it runs in a worker thread. The quality gate and perceptual hash run
later in the Celery worker, not in the API. The request body limit (BodySizeLimitMiddleware)
applies before any of this, while the multipart body is still being received.
"""

import hashlib
import io
import re
import struct
from dataclasses import dataclass
from pathlib import PurePath
from typing import Annotated

import magic
import structlog
from fastapi import Depends, File, UploadFile
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from medlens.core.config import Settings
from medlens.core.errors import PayloadTooLarge, UnsupportedMediaType, ValidationFailed
from medlens.resources import get_app_settings

JPEG_QUALITY = 90
_SNIFF_BYTES = 4096

ALLOWED_TYPES: dict[str, str] = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
_TYPE_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}
_EXTENSION_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_UNDECLARED_TYPES = frozenset({"", "application/octet-stream"})

# Markers of active content. Each is long enough that compressed image data matches it by
# chance far less than once in a million uploads.
_ACTIVE_CONTENT = re.compile(
    rb"<script|</script|<iframe|<object[\s>]|<embed[\s>]|<html[\s>]|<body[\s>]|<!doctype\s+html"
    rb"|<\?php\s|<svg[\s>][^>]{0,200}xmlns|javascript:",
    re.IGNORECASE,
)
_PNG_END = b"IEND\xaeB`\x82"  # IEND chunk type + its fixed CRC
_ZIP_END_OF_CENTRAL_DIRECTORY = b"PK\x05\x06"
_ZIP_EOCD_SEARCH_WINDOW = 22 + 0xFFFF

log = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    """A safe, re-encoded JPEG ready for storage. Deliberately has no field for the original."""

    jpeg: bytes
    width: int
    height: int
    sha256: str  # hex digest of ``jpeg``
    source_type: str  # sniffed type of the upload, e.g. image/png


def process_image(
    data: bytes,
    *,
    declared_type: str | None,
    filename: str | None,
    max_pixels: int,
) -> ValidatedImage:
    """Validate and re-encode an upload already known to be within the size limit."""
    source_type = magic.from_buffer(data[:_SNIFF_BYTES], mime=True)
    if source_type not in ALLOWED_TYPES:
        raise _reject(UnsupportedMediaType("Upload a JPEG, PNG or WebP photo."), "type")
    _check_declared_type(source_type, declared_type, filename)

    if reason := _polyglot_reason(data, source_type):
        error = _invalid_image("The file contains data that is not part of an image.")
        raise _reject(error, reason)

    image = _decode(data, source_type, max_pixels)
    output = io.BytesIO()
    _to_rgb(image).save(output, "JPEG", quality=JPEG_QUALITY)  # no exif=: metadata is dropped
    jpeg = output.getvalue()
    return ValidatedImage(
        jpeg=jpeg,
        width=image.width,
        height=image.height,
        sha256=hashlib.sha256(jpeg).hexdigest(),
        source_type=source_type,
    )


async def validate_image_upload(
    image: Annotated[
        UploadFile,
        File(description="Photo of the pack: JPEG, PNG or WebP, up to 8 MB by default."),
    ],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ValidatedImage:
    """FastAPI dependency for a multipart field named ``image``."""
    limit = settings.app.max_upload_bytes
    try:
        data = await image.read(limit + 1)
    finally:
        await image.close()  # nothing downstream can read the original upload
    if len(data) > limit:
        raise _reject(PayloadTooLarge(f"The image exceeds the limit of {limit} bytes."), "size")
    if not data:
        raise _reject(_invalid_image("The uploaded file is empty."), "empty")
    return await run_in_threadpool(
        process_image,
        data,
        declared_type=image.content_type,
        filename=image.filename,
        max_pixels=settings.app.max_image_pixels,
    )


def _check_declared_type(source_type: str, declared_type: str | None, filename: str | None) -> None:
    declared: set[str] = set()
    content_type = (declared_type or "").split(";")[0].strip().lower()
    if content_type not in _UNDECLARED_TYPES:
        declared.add(_TYPE_ALIASES.get(content_type, content_type))
    extension = PurePath(filename or "").suffix.lower()
    if extension in _EXTENSION_TYPES:
        declared.add(_EXTENSION_TYPES[extension])
    if declared - {source_type}:
        raise _reject(
            UnsupportedMediaType("The file content does not match its declared type or name."),
            "type_mismatch",
        )


def _polyglot_reason(data: bytes, source_type: str) -> str | None:
    if _ACTIVE_CONTENT.search(data):
        return "active_content"
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if _has_zip_trailer(data):
        return "zip"
    if source_type == "image/png" and not data.endswith(_PNG_END):
        return "png_trailing_data"
    if source_type == "image/webp" and _webp_has_trailing_data(data):
        return "webp_trailing_data"
    return None


def _has_zip_trailer(data: bytes) -> bool:
    """A ZIP reader finds the archive from its end record, which must close the file."""
    start = max(0, len(data) - _ZIP_EOCD_SEARCH_WINDOW)
    position = data.rfind(_ZIP_END_OF_CENTRAL_DIRECTORY, start)
    if position == -1 or position + 22 > len(data):
        return False
    comment_length: int = struct.unpack_from("<H", data, position + 20)[0]
    return position + 22 + comment_length == len(data)


def _webp_has_trailing_data(data: bytes) -> bool:
    riff_size: int = struct.unpack_from("<I", data, 4)[0]
    return len(data) > 8 + riff_size + (riff_size & 1)


def _decode(data: bytes, source_type: str, max_pixels: int) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data), formats=[ALLOWED_TYPES[source_type]]) as opened:
            # Image.open only reads the header, so this runs before any pixel is decoded.
            if opened.width * opened.height > max_pixels:
                raise _reject(_invalid_image("The image has too many pixels."), "too_many_pixels")
            if getattr(opened, "n_frames", 1) > 1:
                raise _reject(_invalid_image("Animated images are not accepted."), "animated")
            opened.load()
            return ImageOps.exif_transpose(opened)
    except Image.DecompressionBombError:
        raise _reject(_invalid_image("The image has too many pixels."), "too_many_pixels") from None
    except (OSError, SyntaxError, ValueError):  # Pillow's ways of saying "corrupt"
        raise _reject(_invalid_image("The image could not be decoded."), "corrupt") from None


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        return flattened
    return image.convert("RGB")


def _invalid_image(detail: str) -> ValidationFailed:
    return ValidationFailed(
        detail, errors=[{"loc": ["body", "image"], "msg": detail, "type": "image"}]
    )


def _reject[E: Exception](error: E, reason: str) -> E:
    # Rejections are a security signal (probing, malformed uploads): log the reason, not data.
    log.info("upload.rejected", reason=reason)
    return error


# --------------------------------------------------------------------------- CSV data files
CSV_MAX_BYTES = 2 * 1024 * 1024
_CSV_TYPES = frozenset({"text/plain", "text/csv", "application/csv"})


@dataclass(frozen=True, slots=True)
class CsvUpload:
    data: bytes


async def read_csv_upload(
    file: Annotated[UploadFile, File(description="UTF-8 CSV file, up to 2 MB.")],
) -> CsvUpload:
    """FastAPI dependency for a multipart field named ``file`` holding CSV data: size first,
    then the real type by content (never the client's claim)."""
    try:
        data = await file.read(CSV_MAX_BYTES + 1)
    finally:
        await file.close()
    if len(data) > CSV_MAX_BYTES:
        raise _reject(PayloadTooLarge(f"The file exceeds {CSV_MAX_BYTES} bytes; split it."), "size")
    if not data:
        raise _reject(_invalid_image("The uploaded file is empty."), "empty")
    if magic.from_buffer(data[:_SNIFF_BYTES], mime=True) not in _CSV_TYPES:
        raise _reject(UnsupportedMediaType("Upload a UTF-8 CSV file."), "type")
    return CsvUpload(data)


# --------------------------------------------------------------------------- documents
DOCUMENT_TYPES = frozenset({"application/pdf", "text/plain", "text/markdown", "text/x-markdown"})
# PDF features that run code or carry other files: not needed to read a notice.
_PDF_ACTIVE = re.compile(rb"/(?:JavaScript|JS|Launch|EmbeddedFile|RichMedia|XFA)\b")


@dataclass(frozen=True, slots=True)
class DocumentUpload:
    data: bytes
    media_type: str  # sniffed, never the client's claim


async def read_document_upload(
    file: Annotated[UploadFile, File(description="PDF, plain text or Markdown (UTF-8).")],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> DocumentUpload:
    """FastAPI dependency for a multipart field named ``file`` holding a document: size, the
    real type by content, no active PDF content, and valid UTF-8 for text."""
    limit = settings.knowledge.max_document_bytes
    try:
        data = await file.read(limit + 1)
    finally:
        await file.close()
    if len(data) > limit:
        raise _reject(PayloadTooLarge(f"The document exceeds {limit} bytes."), "size")
    if not data:
        raise _reject(_invalid_document("The uploaded file is empty."), "empty")
    media_type = magic.from_buffer(data[:_SNIFF_BYTES], mime=True)
    if media_type not in DOCUMENT_TYPES:
        raise _reject(UnsupportedMediaType("Upload a PDF, plain text or Markdown file."), "type")
    if media_type == "application/pdf":
        if _PDF_ACTIVE.search(data):
            raise _reject(
                _invalid_document("PDFs with scripts, attachments or forms are not accepted."),
                "pdf_active_content",
            )
    else:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise _reject(_invalid_document("Text files must be UTF-8."), "encoding") from None
        media_type = "text/plain"
    return DocumentUpload(data, media_type)


def _invalid_document(detail: str) -> ValidationFailed:
    return ValidationFailed(
        detail, errors=[{"loc": ["body", "file"], "msg": detail, "type": "document"}]
    )


# The only dependencies that may receive uploaded files (tests/unit/test_uploads.py checks).
VALIDATED_UPLOADS = (validate_image_upload, read_csv_upload, read_document_upload)
