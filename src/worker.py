import base64
import tempfile
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
import runpod
from time import time as now
from multimodal import Multimodal
from schema import get_schema_serverless
from urllib import parse

MODEL = Multimodal()


def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


def download_file(jobId, image_url):
    return download_files_from_urls(jobId, image_url)[0]


@rp_debugger.FunctionTimer
def analyse(job):
    started = now()
    job_input = job["input"]
    result = {}
    if "schema" in job_input:
        result = dict(schema=get_schema_serverless())
    else:
        try:
            params = job_input.get("params", {})
            if "image_raw" in job_input:
                urlEncoded = (
                    job_input["urlEncoded"] if "urlEncoded" in job_input else False
                )
                image = base64_to_tempfile(
                    parse.unquote(job_input["image_raw"])
                    if urlEncoded
                    else job_input["image_raw"]
                )
            elif "image_url" in job_input:
                with rp_debugger.LineTimer("download_step"):
                    image = download_file(job["id"], [job_input["image_url"]])
            else:
                result = dict(error="Must provide either image_raw or image_url")
            if not result:
                with rp_debugger.LineTimer("prediction_step"):
                    result = MODEL.process(image, job_input["prompt"], params)
        except Exception as e:
            result = dict(error=str(e))
    with rp_debugger.LineTimer("cleanup_step"):
        rp_cleanup.clean(["input_objects"])
    return result | {"time": now() - started}


runpod.serverless.start({"handler": analyse})
