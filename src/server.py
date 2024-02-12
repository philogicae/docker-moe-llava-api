from fastapi import FastAPI
from multimodal import Multimodal
from schema import get_schema_server
from time import time as now
from base64 import b64decode
from tempfile import NamedTemporaryFile
from PIL import Image
from urllib import parse
import logging

app = FastAPI(title="moe-llava-api")
logger = logging.getLogger("uvicorn")
MODEL = None


@app.on_event("startup")
def on_startup():
    started = now()
    global MODEL
    MODEL = Multimodal()
    logger.info(f"MODEL: {MODEL.model_path} - Loaded ({now() - started:.2f}s)")


@app.get("/")
async def ping():
    logger.info("API: Ping!")
    return {"status": "Running"}


@app.get("/schema")
async def schema():
    logger.info("API: Schema")
    return {"schema": get_schema_server()}


def base64_to_image(base64_file: str):
    with NamedTemporaryFile(suffix=".tmp", delete=True) as img:
        img.write(b64decode(base64_file))
        return Image.open(img.name).convert("RGB")


def url_to_image(images_url: str):
    return Image.open(images_url).convert("RGB")


@app.post("/vision")
async def vision(job):
    started = now()
    job_input = job["input"]
    output = dict(results=[])
    try:
        to_process = []
        if "data" in job_input:
            output["error"] = "Must provide data[]"
        for item in job_input["data"]:
            images, error = [], ""
            prompt = item.get("prompt", None)
            if not prompt:
                error = "Must provide a prompt"
                to_process.append(error)
            else:
                params = item.get("params", {})
                if "images_base64" in item:
                    images = [
                        base64_to_image(
                            parse.unquote(img) if item.get("urlEncoded", False) else img
                        )
                        for img in item["images_base64"]
                    ]
                elif "images_url" in item:
                    images = [url_to_image(img) for img in item["images_url"]]
                else:
                    error = "Must provide either images_base64 or images_url"
                    to_process.append(error)
                if not error:
                    to_process.append([images, prompt, params])
        if to_process:
            for item in to_process:
                result = ""
                if isinstance(item, str):
                    result = dict(error=item)
                else:
                    started_gen = now()
                    try:
                        result = MODEL.process(*item)
                    except Exception as e:
                        result = dict(error=str(e))
                    output["results"].append(result | {"time": now() - started_gen})
    except Exception as e:
        output["error"] = str(e)
    return output | {"total_time": now() - started}
