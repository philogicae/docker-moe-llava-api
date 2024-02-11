def get_schema_server():
    return {
        "/ping": "check if status",
        "/schema": "return api schema dict",
        "/multimodal": dict(
            expected=dict(
                image="file",
                prompt="str",
                params="inference params",
            ),
            returned=dict(
                text="str",
                error="str",
                time="float",
                schema="dict",
            ),
        ),
    }


def get_schema_serverless():
    return dict(
        expected=dict(
            for_schema=dict(input=dict(schema=True)),
            for_multimodal=dict(
                input=dict(
                    image_raw="base64_image",
                    image_url="str",
                    urlEncoded="bool",
                    prompt="str",
                    params="inference params",
                )
            ),
        ),
        returned=dict(
            text="str",
            error="str",
            time="float",
            schema="dict",
        ),
    )
