
import logging
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder

from backend.validation import (
    validate_upload_metadata,
    validate_file_size,
    MAX_FILE_SIZES,
)
from backend.services import evaluate_resume


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ResumeIQ API",
    description="Backend API for ResumeIQ resume evaluation system",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "ResumeIQ API is running",
        "status": "healthy"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(
    file: UploadFile = File(...),
    evaluation_type: str = Form(...),
    role_name: str | None = Form(None),
):
    # 1. Validate upload metadata before processing.
    is_valid, metadata_or_error = validate_upload_metadata(
        filename=file.filename,
        evaluation_type=evaluation_type,
        role_name=role_name,
    )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=metadata_or_error
        )

    metadata = metadata_or_error
    extension = metadata["extension"]

    # 2. Create a server-generated temporary file.
    temp_file_path = None
    file_size = 0

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            prefix="resumeiq_",
            delete=False,
        ) as temp_file:
            temp_file_path = Path(temp_file.name)

            # 3. Read in chunks and enforce the actual size limit.
            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                file_size += len(chunk)

                is_valid, error_message = validate_file_size(
                    file_size,
                    extension,
                )

                if not is_valid:
                    raise HTTPException(
                        status_code=413,
                        detail=error_message,
                    )

                temp_file.write(chunk)

        # 4. Run the existing evaluation pipeline.
        result = evaluate_resume(
            file_path=str(temp_file_path),
            evaluation_type=metadata["evaluation_type"],
            role_name=metadata["role_name"],
        )

        # 5. Convert Pydantic models and other supported objects to JSON.
        return jsonable_encoder(result)

    except HTTPException:
        raise
    
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        logger.exception("Resume evaluation failed.")

        raise HTTPException(
            status_code=500,
            detail="Resume evaluation failed. Please try again.",
        )

    finally:
        # 6. Always remove the temporary file.
        if temp_file_path is not None:
            try:
                temp_file_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Temporary file cleanup failed.")

        # Close the uploaded file.
        try:
            file.file.close()
        except OSError:
            logger.exception("Uploaded file cleanup failed.")
