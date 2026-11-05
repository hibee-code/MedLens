"""Process-wide setup shared by every entrypoint: API, Celery worker and beat, seed, scripts."""

from PIL import Image

from medlens.core.config import Settings
from medlens.core.crypto import configure_crypto
from medlens.core.logging import configure_logging
from medlens.core.telemetry import configure_sentry, configure_tracing


def configure_process(settings: Settings, component: str = "cli") -> None:
    """``component``: api, worker or cli; names the service in traces and error reports."""
    configure_logging(settings)
    configure_crypto(settings.crypto)
    configure_tracing(settings, component)
    configure_sentry(settings, component)
    # Pillow's own decompression-bomb guard; core/uploads.py also checks dimensions explicitly.
    Image.MAX_IMAGE_PIXELS = settings.app.max_image_pixels
