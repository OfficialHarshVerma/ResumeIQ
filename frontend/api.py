
import requests


API_URL = "http://127.0.0.1:8000"


def evaluate_resume(
    uploaded_file,
    evaluation_type,
    role_name=None
):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/octet-stream"
        )
    }

    data = {
        "evaluation_type": evaluation_type
    }

    if role_name:
        data["role_name"] = role_name

    response = requests.post(
        f"{API_URL}/evaluate",
        files=files,
        data=data,
        timeout=300
    )

    response.raise_for_status()

    return response.json()
