
from pathlib import Path


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


# Keep this configurable until we verify the exact limit
# configured in extraction.py.
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


# =========================================================
# FILE VALIDATION
# =========================================================

def validate_upload(uploaded_file):
    """
    Perform preliminary frontend validation on an uploaded file.

    Returns:
        (True, None)       -> Validation successful
        (False, message)   -> Validation failed
    """

    # -----------------------------------------------------
    # 1. CHECK FILE EXISTENCE
    # -----------------------------------------------------

    if uploaded_file is None:
        return False, "Please upload a resume."

    # -----------------------------------------------------
    # 2. CHECK FILE NAME
    # -----------------------------------------------------


    # -----------------------------------------------------
    # 2. CHECK FILE NAME
    # -----------------------------------------------------

    filename = uploaded_file.name

    if not filename or not filename.strip():
        return False, "Invalid filename."

    # Maximum filename length
    if len(filename) > 255:
        return False, "Filename is too long. Maximum length is 255 characters."

    # Reject null bytes and control characters
    if any(ord(char) < 32 or ord(char) == 127 for char in filename):
        return False, "Filename contains invalid characters."

    # Reject path separators
    if "/" in filename or "\\" in filename:
        return False, "Filename must not contain directory paths."

    # Extract extension
    extension = Path(filename).suffix.lower()

    if not extension:
        return False, "Filename must have a file extension."


    # -----------------------------------------------------
    # 3. CHECK FILE EXTENSION
    # -----------------------------------------------------

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))

        return False, (
            f"Unsupported file format. "
            f"Allowed formats: {allowed}"
        )

    # -----------------------------------------------------
    # 4. CHECK EMPTY FILE
    # -----------------------------------------------------

    if uploaded_file.size == 0:
        return False, "The uploaded file is empty."

    # -----------------------------------------------------
    # 5. CHECK FILE SIZE
    # -----------------------------------------------------

    maximum_size = MAX_FILE_SIZES[extension]

    if uploaded_file.size > maximum_size:

        maximum_size_mb = maximum_size / (1024 * 1024)

        return False, (
            f"File size exceeds the maximum allowed "
            f"limit of {maximum_size_mb:g} MB "
            f"for {extension} files."
        )

    
    # -----------------------------------------------------
    # 6. VALIDATION SUCCESSFUL
    # -----------------------------------------------------

    return True, None