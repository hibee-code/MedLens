"""Process-wide setup shared by every entrypoint: API, Celery worker and beat, seed, scripts."""

from PIL import Image

from medlens.core.config import Settings
from medlens.core.crypto import configure_crypto
from medlens.core.logging import configure_logging


def configure_process(settings: Settings) -> None:
    configure_logging(settings)
    configure_crypto(settings.crypto)
    # Pillow's own decompression-bomb guard; core/uploads.py also checks dimensions explicitly.
    Image.MAX_IMAGE_PIXELS = settings.app.max_image_pixels
