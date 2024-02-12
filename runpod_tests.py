import runpod
import base64
from rich import print
from dotenv import load_dotenv
from os import getenv
from json import dumps

load_dotenv()

runpod.api_key = getenv("RUNPOD_API_KEY")
endpoint_id = getenv("RUNPOD_ENDPOINT_ID")
endpoint = runpod.Endpoint(endpoint_id)
TIMEOUT = 120


def sync_call(data):
    print("Start sync call...")
    return endpoint.run_sync({"input": data})


def async_call(data):
    print("Start async call...")
    job = endpoint.run({"input": data})
    print(job.status())
    return lambda: job.output(TIMEOUT)


def load_img(filename):
    with open("input/" + filename, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_output(filename, result):
    with open("output/" + filename + ".txt", "w", encoding="utf-8") as f:
        f.write(dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    data = dict(
        data=[
            dict(
                images_base64=[
                    load_img(img) for img in input("Filenames: ").split(" ")
                ],
                prompt=input("Prompt: "),
            )
        ]
    )
    # result = sync_call(data)
    result = async_call(data)()
    print(result)
    save_output("test", result)
