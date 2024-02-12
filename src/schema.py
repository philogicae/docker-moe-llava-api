def get_schema_server():
    return {
        "/ping": "check if status",
        "/schema": "return api schema dict",
        "/vision": dict(
            expected=dict(
                input=dict(
                    data=[
                        dict(
                            images_base64="[image_base64, ...]",
                            images_url="[image_url, ...]",
                            urlEncoded="bool",
                            prompt="str",
                            params="inference params",
                        ),
                        "...",
                    ]
                ),
            ),
            returned=dict(
                results=[
                    dict(
                        text="str",
                        error="str",
                        time="float",
                    ),
                    "...",
                ],
                error="str",
                schema="dict",
                total_time="float",
            ),
        ),
    }


def get_schema_serverless():
    return dict(
        expected=dict(
            for_schema=dict(input=dict(schema=True)),
            for_vision=dict(
                input=dict(
                    data=[
                        dict(
                            images_base64="[image_base64, ...]",
                            images_url="[image_url, ...]",
                            urlEncoded="bool",
                            prompt="str",
                            params="inference params",
                        ),
                        "...",
                    ]
                ),
            ),
        ),
        returned=dict(
            results=[
                dict(
                    text="str",
                    error="str",
                    time="float",
                ),
                "...",
            ],
            error="str",
            schema="dict",
            total_time="float",
        ),
    )
