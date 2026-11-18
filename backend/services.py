
from generator import (
    generate_general,
    generate_role_specific
)


from generator import (
    generate_general,
    generate_role_specific
)

from benchmarks import get_role_guide


def evaluate_resume(
    file_path: str,
    evaluation_type: str,
    role_name: str | None = None
):
    if evaluation_type == "general":
        return generate_general(file_path)

    if evaluation_type == "role_specific":

        role_guide = get_role_guide(role_name)

        if role_guide is None:
            raise ValueError(
                f"Unsupported role: {role_name}"
            )

        return generate_role_specific(
            file_path=file_path,
            role_name=role_name
        )

    raise ValueError("Unsupported evaluation type.")



def evaluate_resume(
    file_path: str,
    evaluation_type: str,
    role_name: str | None = None
):
    """
    Route the uploaded resume to the appropriate evaluation pipeline.
    """

    if evaluation_type == "general":
        return generate_general(file_path)

    if evaluation_type == "role_specific":
        return generate_role_specific(
            file_path=file_path,
            role_name=role_name
        )

    raise ValueError("Unsupported evaluation type.")




