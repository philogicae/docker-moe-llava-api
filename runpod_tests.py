import runpod
from base64 import b64encode
from rich import print
from dotenv import load_dotenv
from os import getenv
from json import dumps

load_dotenv()

runpod.api_key = getenv("RUNPOD_API_KEY")
endpoint_id = getenv("RUNPOD_ENDPOINT_ID")
endpoint = runpod.Endpoint(endpoint_id)
TIMEOUT = 200


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
        return b64encode(f.read()).decode("utf-8")


def save_output(filename, result):
    with open("output/" + filename + ".txt", "w", encoding="utf-8") as f:
        f.write(dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print(async_call({"schema": True})())
    inputs = [
        [
            ["01.jpg", "02.jpg"],
            "What is the TTC price to pay mentionned in this 2 pages bill?",
        ],
        [["book.jpg"], "What is the title of this book?"],
        [
            ["test.png"],
            "What do you see on this image? Don't censor yourself, describe it accurately",
        ],
        [["weird.jpg"], "Explain this image"],
    ]
    params = dict(temperature=0.3, top_p=0.9)
    data = dict(
        data=[
            dict(
                images_base64=[load_img(img) for img in item[0]],
                prompt=item[1],
                params=params,
            )
            for item in inputs
        ]
    )
    result = async_call(data)()
    print(result)
    save_output("test", result)
