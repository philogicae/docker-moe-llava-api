from base64 import b64decode
from tempfile import NamedTemporaryFile
from PIL import Image


def base64_to_image(base64_file: str):
    with NamedTemporaryFile(suffix=".tmp", delete=True) as img:
        img.write(b64decode(base64_file))
        return Image.open(img.name).convert("RGB")


def url_to_image(images_url: str):
    return Image.open(images_url).convert("RGB")
