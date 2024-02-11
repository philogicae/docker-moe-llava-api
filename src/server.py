from fastapi import FastAPI, UploadFile, HTTPException
from multimodal import Multimodal
from schema import get_schema_server
from io import BytesIO
from time import time as now
from json import loads
import logging

app = FastAPI(title="faster-whisper-fr-api")
logger = logging.getLogger("uvicorn")
model = None


@app.on_event("startup")
def on_startup():
    started = now()
    global model
    model = Multimodal()
    logger.info(f"MODEL: {model.model_path} - Loaded ({now() - started:.2f}s)")


@app.get("/")
async def ping():
    logger.info("API: Ping!")
    return {"status": "Running"}


@app.get("/schema")
async def schema():
    logger.info("API: Schema")
    return {"schema": get_schema_server()}


@app.post("/multimodal")
async def multimodal(image: UploadFile, prompt: str, params=None):
    started = now()
    logger.info(f"FILE: {image.filename} - Start processing...")
    try:
        params = (
            (params if isinstance(params, dict) else loads(params)) if params else {}
        )
        result = model.process(BytesIO(await image.read()), prompt, params)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error during transcription: " + str(e)
        )
    stop = now() - started
    logger.info(f"FILE: {image.filename} - Processed ({stop/60:.0f}m{stop%60:.2f}s)")
    return result | {"time": stop}
