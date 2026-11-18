
from pathlib import Path
from typing import Optional


# =========================================================
# CONFIGURATION
# =========================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".doc",
    ".rtf",
    ".jpg",
    ".jpeg",
    ".png",
}

MAX_FILE_SIZES = {
    ".pdf": 5 * 1024 * 1024,
    ".docx": 5 * 1024 * 1024,
    ".txt": 5 * 1024 * 1024,
    ".doc": 20 * 1024 * 1024,
    ".rtf": 10 * 1024 * 1024,
    ".jpg": 10 * 1024 * 1024,
    ".jpeg": 10 * 1024 * 1024,
    ".png": 10 * 1024 * 1024,
}

MAX_FILENAME_LENGTH = 255
MAX_ROLE_LENGTH = 100

ALLOWED_EVALUATION_TYPES = {
    "general",
    "role_specific",
}


# =========================================================
# FILENAME VALIDATION
# =========================================================

def validate_filename(filename: Optional[str]):

    if not isinstance(filename, str) or not filename.strip():
        return False, "Filename is required."

    if len(filename) > MAX_FILENAME_LENGTH:
        return False, "Filename is too long."

    if any(ord(char) < 32 or ord(char) == 127 for char in filename):
        return False, "Filename contains invalid characters."

    if "/" in filename or "\\" in filename:
        return False, "Path separators are not allowed."

    if "\x00" in filename:
        return False, "Filename contains a null byte."

    extension = Path(filename).suffix.lower()

    if not extension:
        return False, "Filename must have an extension."

    if extension not in ALLOWED_EXTENSIONS:
        return False, "Unsupported file format."

    return True, extension


# =========================================================
# FILE SIZE VALIDATION
# =========================================================

def validate_file_size(file_size: int, extension: str):

    if not isinstance(file_size, int) or file_size < 0:
        return False, "Invalid file size."

    if file_size == 0:
        return False, "Uploaded file is empty."

    extension = extension.lower()

    maximum_size = MAX_FILE_SIZES.get(extension)

    if maximum_size is None:
        return False, "Unsupported file format."

    if file_size > maximum_size:

        maximum_size_mb = maximum_size / (1024 * 1024)

        return False, (
            f"File exceeds the maximum allowed size "
            f"of {maximum_size_mb:g} MB."
        )

    return True, None


# =========================================================
# EVALUATION TYPE VALIDATION
# =========================================================

def validate_evaluation_type(evaluation_type: str):

    if not isinstance(evaluation_type, str):
        return False, "Invalid evaluation type."

    evaluation_type = evaluation_type.strip().lower()

    if evaluation_type not in ALLOWED_EVALUATION_TYPES:
        return False, "Unsupported evaluation type."

    return True, evaluation_type


# =========================================================
# ROLE VALIDATION
# =========================================================

def validate_role_name(
    role_name: Optional[str],
    evaluation_type: str
):

    if evaluation_type == "general":
        return True, None

    if evaluation_type != "role_specific":
        return False, "Invalid evaluation type."

    if not isinstance(role_name, str) or not role_name.strip():
        return False, "Target role is required."

    role_name = role_name.strip()

    if len(role_name) > MAX_ROLE_LENGTH:
        return False, "Target role is too long."

    if any(ord(char) < 32 or ord(char) == 127 for char in role_name):
        return False, "Target role contains invalid characters."

    return True, role_name


# =========================================================
# COMPLETE METADATA VALIDATION
# =========================================================

def validate_upload_metadata(
    filename: Optional[str],
    evaluation_type: str,
    role_name: Optional[str] = None
):
    """
    Validate upload metadata before reading or processing
    the uploaded file.

    Returns:
        (True, metadata_dict)
        (False, error_message)
    """

    # 1. Filename
    valid, extension_or_error = validate_filename(filename)

    if not valid:
        return False, extension_or_error

    extension = extension_or_error

    # 2. Evaluation type
    valid, evaluation_or_error = validate_evaluation_type(
        evaluation_type
    )

    if not valid:
        return False, evaluation_or_error

    evaluation_type = evaluation_or_error

    # 3. Role
    valid, role_or_error = validate_role_name(
        role_name,
        evaluation_type
    )

    if not valid:
        return False, role_or_error

    return True, {
        "filename": filename,
        "extension": extension,
        "evaluation_type": evaluation_type,
        "role_name": role_or_error,
    }
