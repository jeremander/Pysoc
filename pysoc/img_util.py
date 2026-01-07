import base64
from io import BytesIO
from pathlib import Path
from typing import IO

from PIL import ImageOps
import PIL.Image
from PIL.Image import Image


THUMBNAIL_WIDTH = 200


def load_image(path: str | Path | IO[bytes]) -> Image:
    """Loads an image and orients it according to its EXIF orientiation, if present."""
    img = PIL.Image.open(path)
    return ImageOps.exif_transpose(img)

def img_from_bytes(b: bytes) -> Image:
    """Loads an image from raw bytes."""
    return load_image(BytesIO(b))

def img_to_base64(img: Image, format: str ='jpeg') -> str:
    """Converts an image to a base64-encoded string."""
    bio = BytesIO()
    img.save(bio, format=format)
    return base64.b64encode(bio.getvalue()).decode()

def img_from_base64(s: str) -> Image:
    """Loads an image from a base64-encoded string."""
    return img_from_bytes(base64.b64decode(s))

def make_border(img: Image, border_frac: float = 0.10) -> Image:
    """Puts the image in a box with a border.
    Does not change the image aspect ratio."""
    size = img.size
    c = 1 + 2 * border_frac
    new_size = (int(c * size[0]), int(c * size[1]))
    border = min((new_size[0] - size[0]) // 2, (new_size[1] - size[1]) // 2)
    return ImageOps.expand(img, border, fill='white')

def make_thumbnail(img: Image, width: int = THUMBNAIL_WIDTH) -> Image:
    """Resizes the image into a square thumbnail.
    This can stretch the image."""
    size = img.size
    ratio = width / size[0]
    new_size = (int(size[0] * ratio), int(size[1] * ratio))
    return img.resize(new_size)
