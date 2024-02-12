from runpod import serverless
from runpod.serverless.utils import rp_cleanup, rp_debugger
from time import time as now
from vision import Vision
from schema import get_schema_serverless
from prepare import base64_to_image, url_to_image
from urllib import parse

MODEL = Vision()


@rp_debugger.FunctionTimer
def worker(job):
    started = now()
    job_input = job["input"]
    output = dict(results=[])
    if "schema" in job_input:
        output = dict(schema=get_schema_serverless())
    else:
        try:
            to_process = []
            if "data" not in job_input:
                output["error"] = "Must provide data[]"
            else:
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
                                    parse.unquote(img)
                                    if item.get("urlEncoded", False)
                                    else img
                                )
                                for img in item["images_base64"]
                            ]
                        elif "images_url" in item:
                            with rp_debugger.LineTimer("download_step"):
                                images = [
                                    url_to_image(img) for img in item["images_url"]
                                ]
                        else:
                            error = "Must provide either images_base64 or images_url"
                            to_process.append(error)
                        if not error:
                            to_process.append([images, prompt, params])
                if to_process:
                    with rp_debugger.LineTimer("prediction_step"):
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
                                output["results"].append(
                                    result | {"time": now() - started_gen}
                                )
        except Exception as e:
            output["error"] = str(e)
    with rp_debugger.LineTimer("cleanup_step"):
        rp_cleanup.clean(["input_objects"])
    return output | {"total_time": now() - started}


serverless.start({"handler": worker})
