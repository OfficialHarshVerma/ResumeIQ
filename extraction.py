import os
import time
import shutil
import hashlib
import zipfile
import zlib
import codecs
import warnings
import subprocess
import tempfile
import multiprocessing as mp
import threading
from pathlib import Path
from io import BytesIO
import magic
import psutil
import pymupdf
import pytesseract
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from PIL import (
    Image,
    ImageFile,
    Image as PILImage,
    UnidentifiedImageError
)
from lxml import etree
from striprtf.striprtf import rtf_to_text
import win32api
import win32job
from database.medallion_db import (
    initialize_database,
    get_bronze_by_fingerprint,
    save_bronze,
    process_silver,
    process_gold
)

# =========================================================
# 1. GLOBAL CONFIGURATION
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".doc",
    ".rtf", ".jpg", ".jpeg", ".png"
}

SUPPORTED_FILE_TYPES = {
    "pdf", "docx", "txt", "doc",
    "rtf", "jpeg", "png"
}

MAX_EXTRACTED_TEXT_CHARS = 1_000_000

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

MIME_TYPES = {
    "pdf": {"application/pdf"},

    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/x-zip",
        "application/octet-stream",
    },

    "doc": {
        "application/msword",
        "application/x-ole-storage",
        "application/CDFV2",
    },

    "txt": {"text/plain"},

    "rtf": {
        "application/rtf",
        "text/rtf",
        "application/x-rtf",
        "text/plain",
    },

    "jpeg": {"image/jpeg"},
    "png": {"image/png"},
}


# =========================================================
# 2. FILE VALIDATION
# =========================================================

def validate_file_path_and_extension(file_path):

    file_path = Path(file_path).resolve(strict=True)

    if not file_path.is_file():
        raise ValueError("The specified path is not a regular file.")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file format.")

    return file_path


def validate_file_size(file_path):

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension not in MAX_FILE_SIZES:
        raise ValueError("Unsupported file extension.")

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise ValueError("Uploaded file is empty.")

    maximum_size = MAX_FILE_SIZES[extension]

    if file_size > maximum_size:
        raise ValueError(
            f"File exceeds the maximum allowed size of "
            f"{maximum_size // (1024 * 1024)} MB."
        )

    return True


# =========================================================
# 3. FILE TYPE DETECTION
# =========================================================

def detect_file_type(file_path):

    file_path = Path(file_path)

    with open(file_path, "rb") as file:
        header = file.read(16)

    if header.startswith(b"%PDF-"):
        return "pdf"

    if header.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        return "doc"

    if header.startswith(bytes.fromhex("FFD8FF")):
        return "jpeg"

    if header.startswith(bytes.fromhex("89504E470D0A1A0A")):
        return "png"

    if header.lstrip().startswith(b"{\\rtf"):
        return "rtf"

    if zipfile.is_zipfile(file_path):

        try:
            with zipfile.ZipFile(file_path, "r") as archive:
                names = set(archive.namelist())

                if "word/document.xml" in names:
                    return "docx"

        except (zipfile.BadZipFile, OSError):
            pass

    # Text detection supports UTF-8 and Windows-1252.
    # Null bytes are rejected because they often indicate binary data.
    with open(file_path, "rb") as file:
        content = file.read(MAX_FILE_SIZES[".txt"] + 1)

    if b"\x00" in content:
        raise ValueError("File format could not be identified.")

    try:
        content.decode("utf-8-sig", errors="strict")
        return "txt"
    except UnicodeDecodeError:
        pass

    try:
        content.decode("cp1252", errors="strict")
        return "txt"
    except UnicodeDecodeError:
        pass

    raise ValueError(
        "File format could not be identified or is unsupported."
    )


def validate_actual_file_type(file_path, detected_type):

    extension = Path(file_path).suffix.lower().lstrip(".")

    if extension in {"jpg", "jpeg"}:
        extension = "jpeg"

    if detected_type != extension:
        raise ValueError(
            f"File extension mismatch: expected {extension}, "
            f"detected {detected_type}."
        )

    return True


# =========================================================
# 4. MIME VALIDATION
# =========================================================

def detect_mime_type(file_path):

    return magic.from_file(str(file_path), mime=True)


def validate_mime_type(file_path, detected_type):

    actual_mime = detect_mime_type(file_path)

    expected_mimes = MIME_TYPES.get(detected_type)

    if expected_mimes is None:
        raise ValueError("Unsupported detected file type.")

    if actual_mime not in expected_mimes:
        raise ValueError(
            f"MIME type mismatch: expected one of "
            f"{expected_mimes}, got {actual_mime}."
        )

    return actual_mime


# =========================================================
# 5. COMBINED VALIDATION
# =========================================================

def validate_resume_file(file_path):

    file_path = validate_file_path_and_extension(file_path)

    validate_file_size(file_path)

    detected_type = detect_file_type(file_path)

    validate_actual_file_type(file_path, detected_type)

    actual_mime = validate_mime_type(
        file_path,
        detected_type
    )

    print("Validation completed.")

    return {
        "file_path": str(file_path),
        "file_type": detected_type,
        "mime_type": actual_mime,
        "size_bytes": file_path.stat().st_size,
        "is_valid": True,
    }


CLAMSCAN_PATH = Path(
    r"C:\Program Files\ClamAV\clamscan.exe"
)

CLAMAV_DATABASE = (
    Path(os.environ["LOCALAPPDATA"])
    / "ClamAV"
    / "database"
)

CLAMAV_TIMEOUT = 120


def scan_with_clamav(file_path):

    if not CLAMSCAN_PATH.is_file():
        raise RuntimeError(
            "ClamAV scanner executable is unavailable."
        )

    if not CLAMAV_DATABASE.is_dir():
        raise RuntimeError(
            "ClamAV virus database is unavailable."
        )

    command = [
        str(CLAMSCAN_PATH),
        f"--database={CLAMAV_DATABASE}",
        "--infected",
        "--no-summary",
        str(file_path),
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CLAMAV_TIMEOUT,
            shell=False,
            errors="replace",
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Antivirus scan timed out."
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            "Unable to execute ClamAV scanner."
        ) from exc

    if result.returncode == 0:
        return True

    if result.returncode == 1:
        raise ValueError(
            "Malicious content detected. File rejected."
        )

    raise RuntimeError(
        f"ClamAV scan failed. "
        f"Exit code: {result.returncode}. "
        f"Details: {result.stderr.strip() or result.stdout.strip()}"
    )



# =========================================================
# 6. FINGERPRINTING
# =========================================================

def generate_fingerprint(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# =========================================================
# 7. PDF EXTRACTION
# =========================================================

MAX_PDF_PAGES = 50
MAX_PDF_PROCESSING_SECONDS = 45

MAX_PDF_OCR_PAGES = 10
PDF_OCR_DPI = 200

MAX_PDF_PAGE_WIDTH = 1440
MAX_PDF_PAGE_HEIGHT = 1440
MAX_PDF_PAGE_AREA = 1_500_000

MAX_TABLE_PAGES = 20
MAX_TABLES_PER_PAGE = 10
MAX_TABLE_CELLS_PER_PAGE = 200
MAX_TOTAL_TABLE_CELLS = 1000

MAX_PDF_EXTRACTED_CHARS = 200_000
MAX_PDF_PAGE_TEXT_CHARS = 100_000

MAX_PDF_IMAGES = 100
MAX_PDF_IMAGE_WIDTH = 5000
MAX_PDF_IMAGE_HEIGHT = 5000
MAX_PDF_IMAGE_PIXELS = 15_000_000
MAX_TOTAL_PDF_IMAGE_PIXELS = 50_000_000

MAX_PDF_OCR_TEXT_CHARS = 200_000
MAX_TABLE_TEXT_CHARS = 100_000

TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"


def check_pdf_deadline(start_time):

    if time.monotonic() - start_time > MAX_PDF_PROCESSING_SECONDS:
        raise TimeoutError("PDF exceeded the processing time limit.")


def validate_pdf_document(docs):

    if not docs.is_pdf:
        raise ValueError("Invalid PDF document.")

    if docs.needs_pass:
        raise ValueError("Password-protected PDFs are unsupported.")

    if docs.is_repaired:
        raise ValueError("PDF appears corrupted and required repair.")

    if docs.page_count == 0:
        raise ValueError("PDF contains no pages.")

    if docs.page_count > MAX_PDF_PAGES:
        raise ValueError("PDF exceeds the maximum page count.")

    return True


def validate_pdf_images(page, seen_images, total_image_pixels):

    for image_info in page.get_images(full=True):

        xref = image_info[0]

        if xref in seen_images:
            continue

        width = image_info[2]
        height = image_info[3]

        if width <= 0 or height <= 0:
            raise ValueError("PDF contains an invalid embedded image.")

        if (
            width > MAX_PDF_IMAGE_WIDTH
            or height > MAX_PDF_IMAGE_HEIGHT
            or width * height > MAX_PDF_IMAGE_PIXELS
        ):
            raise ValueError("PDF contains an oversized embedded image.")

        total_image_pixels += width * height

        if total_image_pixels > MAX_TOTAL_PDF_IMAGE_PIXELS:
            raise ValueError("PDF exceeds its total image pixel limit.")

        seen_images.add(xref)

        if len(seen_images) > MAX_PDF_IMAGES:
            raise ValueError("PDF contains too many embedded images.")

    return total_image_pixels


def pdf_table_extractor(
    page,
    remaining_cells,
    remaining_chars,
    start_time,
):

    extracted_tables = []
    cells_processed = 0
    tables_processed = 0
    stop_extraction = False

    check_pdf_deadline(start_time)

    table_finder = page.find_tables()

    check_pdf_deadline(start_time)

    for table in table_finder.tables:

        if tables_processed >= MAX_TABLES_PER_PAGE:
            break

        tables_processed += 1

        for row in table.rows:

            check_pdf_deadline(start_time)

            available_cells = min(
                MAX_TABLE_CELLS_PER_PAGE - cells_processed,
                remaining_cells ,
            )

            if available_cells <= 0 or len(row.cells) > available_cells:
                stop_extraction = True
                break

            extracted_row = []

            for cell_bbox in row.cells:

                cells_processed += 1

                if cell_bbox is None:
                    extracted_row.append("")
                    continue

                words = page.get_text(
                    "words",
                    clip=pymupdf.Rect(cell_bbox),
                    sort=True,
                )

                cell_text = " ".join(word[4] for word in words)
                extracted_row.append(cell_text.strip())

            cells = [cell for cell in extracted_row if cell]

            if cells:

                row_text = " | ".join(cells)

                # Include the newline separator in the allowance.
                row_size = len(row_text) + (
                    1 if extracted_tables else 0
                )

                if row_size > remaining_chars:
                    stop_extraction = True
                    break

                extracted_tables.append(row_text)
                remaining_chars -= row_size

            if (
                cells_processed >= MAX_TABLE_CELLS_PER_PAGE
                or cells_processed >= remaining_cells
                or remaining_chars <= 0
            ):
                stop_extraction = True
                break

        if stop_extraction:
            break

    return "\n".join(extracted_tables), cells_processed


def load_pdf(file_path):

    start_time = time.monotonic()

    extracted_text = []

    total_chars = 0
    ocr_pages = 0
    total_ocr_chars = 0

    table_pages = 0
    total_table_cells = 0
    total_table_chars = 0

    seen_images = set()
    total_image_pixels = 0

    docs = None

    try:

        check_pdf_deadline(start_time)

        docs = pymupdf.open(file_path)

        validate_pdf_document(docs)

        for page_number, page in enumerate(docs):

            check_pdf_deadline(start_time)

            page_rect = page.rect

            width = page_rect.width
            height = page_rect.height

            if (
                width <= 0
                or height <= 0
                or width > MAX_PDF_PAGE_WIDTH
                or height > MAX_PDF_PAGE_HEIGHT
                or width * height > MAX_PDF_PAGE_AREA
            ):
                raise ValueError(
                    f"PDF page {page_number + 1} has unacceptable dimensions."
                )

            total_image_pixels = validate_pdf_images(
                page,
                seen_images,
                total_image_pixels,
            )

            check_pdf_deadline(start_time)

            text = page.get_text(sort=True)

            check_pdf_deadline(start_time)

            if len(text) > MAX_PDF_PAGE_TEXT_CHARS:
                raise ValueError(
                    f"PDF page {page_number + 1} contains excessive text."
                )

            if len(text.strip()) < 20:

                if ocr_pages >= MAX_PDF_OCR_PAGES:
                    raise ValueError(
                        "PDF exceeds the maximum number of OCR pages."
                    )

                ocr_pages += 1

                check_pdf_deadline(start_time)

                text_page = page.get_textpage_ocr(
                    language="eng",
                    dpi=PDF_OCR_DPI,
                    full=True,
                    tessdata=TESSDATA_PATH,
                )

                check_pdf_deadline(start_time)

                text = page.get_text(
                    textpage=text_page,
                    sort=True,
                )

                if len(text) > MAX_PDF_PAGE_TEXT_CHARS:
                    raise ValueError(
                        "PDF OCR output exceeds the page text limit."
                    )

                total_ocr_chars += len(text)

                if total_ocr_chars > MAX_PDF_OCR_TEXT_CHARS:
                    raise ValueError(
                        "PDF exceeds the maximum OCR output size."
                    )

            extracted_text.append(text)
            total_chars += len(text)

            if total_chars > MAX_PDF_EXTRACTED_CHARS:
                raise ValueError("PDF extracted text exceeds the allowed size.")

            if (
                table_pages < MAX_TABLE_PAGES
                and total_table_cells < MAX_TOTAL_TABLE_CELLS
                and total_table_chars < MAX_TABLE_TEXT_CHARS
            ):

                remaining_cells = (
                    MAX_TOTAL_TABLE_CELLS - total_table_cells
                )

                remaining_chars = (
                    MAX_TABLE_TEXT_CHARS - total_table_chars
                )

                table_pages += 1

                table_text, cells_used = pdf_table_extractor(
                    page,
                    remaining_cells,
                    remaining_chars,
                    start_time,
                )

                total_table_cells += cells_used
                total_table_chars += len(table_text)

                if table_text:

                    total_chars += len(table_text)

                    if total_chars > MAX_PDF_EXTRACTED_CHARS:
                        raise ValueError(
                            "PDF extracted text exceeds the allowed size."
                        )

                    extracted_text.append(table_text)

            check_pdf_deadline(start_time)

        return "\n".join(extracted_text)

    except (ValueError, TimeoutError):
        raise

    except (
        pymupdf.FileDataError,
        RuntimeError,
        OSError,
        zlib.error,
    ):
        raise ValueError(
            "PDF is corrupted or could not be processed safely."
        ) from None

    finally:

        if docs is not None:
            docs.close()


# =========================================================
# 8. DOCX EXTRACTION
# =========================================================

MAX_DOCX_PROCESSING_SECONDS = 60
MAX_DOCX_EXTRACTED_CHARS = 500_000

MAX_EMBEDDED_IMAGES = 50

MAX_DOCX_IMAGE_PIXELS = 25_000_000
MAX_TOTAL_DOCX_IMAGE_PIXELS = 100_000_000

OCR_TIMEOUT_SECONDS = 10

MAX_ZIP_ENTRIES = 1000
MAX_TOTAL_UNCOMPRESSED_SIZE = 50 * 1024 * 1024
MAX_SINGLE_FILE_SIZE = 10 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100

ImageFile.LOAD_TRUNCATED_IMAGES = False


def check_docx_deadline(start_time):

    if time.monotonic() - start_time > MAX_DOCX_PROCESSING_SECONDS:
        raise TimeoutError("DOCX processing time limit exceeded.")


def validate_docx_archive(file_path):

    try:

        with zipfile.ZipFile(file_path, "r") as archive:

            entries = archive.infolist()

            if len(entries) > MAX_ZIP_ENTRIES:
                raise ValueError("DOCX contains too many internal files.")

            total_size = 0
            actual_total_size = 0
            seen_names = set()

            for entry in entries:

                if entry.filename in seen_names:
                    raise ValueError("DOCX contains duplicate ZIP entries.")

                seen_names.add(entry.filename)

                if entry.is_dir():
                    continue

                if entry.flag_bits & 0x1:
                    raise ValueError("Encrypted DOCX archives are unsupported.")

                if entry.file_size > MAX_SINGLE_FILE_SIZE:
                    raise ValueError(
                        f"DOCX contains an oversized internal file: "
                        f"{entry.filename}"
                    )

                total_size += entry.file_size

                if total_size > MAX_TOTAL_UNCOMPRESSED_SIZE:
                    raise ValueError("DOCX expands beyond the allowed size.")

                if entry.file_size > 1024 * 1024:

                    if entry.compress_size == 0:
                        raise ValueError("Suspicious ZIP compression metadata.")

                    ratio = entry.file_size / entry.compress_size

                    if ratio > MAX_COMPRESSION_RATIO:
                        raise ValueError(
                            "DOCX contains a suspicious compression ratio."
                        )

            required_files = {
                "[Content_Types].xml",
                "_rels/.rels",
                "word/document.xml",
            }

            if not required_files.issubset(seen_names):
                raise ValueError("Invalid DOCX archive structure.")

            # Read every member in bounded chunks to verify CRC and
            # enforce actual decompressed size.
            for entry in entries:

                if entry.is_dir():
                    continue

                with archive.open(entry, "r") as internal_file:

                    while True:

                        chunk = internal_file.read(64 * 1024)

                        if not chunk:
                            break

                        actual_total_size += len(chunk)

                        if actual_total_size > MAX_TOTAL_UNCOMPRESSED_SIZE:
                            raise ValueError(
                                "DOCX exceeds the actual decompressed size limit."
                            )

            parser = etree.XMLParser(
                resolve_entities=False,
                no_network=True,
                load_dtd=False,
                huge_tree=False,
            )

            for entry in entries:

                if entry.is_dir():
                    continue

                if entry.filename.endswith((".xml", ".rels")):

                    with archive.open(entry, "r") as xml_file:
                        etree.parse(xml_file, parser)

            return True

    except ValueError:
        raise

    except zipfile.BadZipFile:
        raise ValueError("Invalid or corrupted DOCX ZIP archive.") from None

    except etree.XMLSyntaxError:
        raise ValueError("DOCX contains malformed XML.") from None

    except (
        RuntimeError,
        zlib.error,
        EOFError,
        NotImplementedError,
    ):
        raise ValueError(
            "Unable to safely inspect DOCX archive."
        ) from None


def extract_docx_images(doc, start_time):

    extracted_text = []
    processed_images = set()

    image_count = 0
    total_image_pixels = 0

    for rel in doc.part.rels.values():

        check_docx_deadline(start_time)

        if rel.reltype != RT.IMAGE:
            continue

        # Check ZIP metadata before loading the image blob.
        target_part = rel.target_part
        image_bytes = target_part.blob

        if len(image_bytes) > MAX_SINGLE_FILE_SIZE:
            raise ValueError("Embedded DOCX image exceeds the size limit.")

        image_hash = hashlib.sha256(image_bytes).hexdigest()

        if image_hash in processed_images:
            continue

        processed_images.add(image_hash)
        image_count += 1

        if image_count > MAX_EMBEDDED_IMAGES:
            raise ValueError("DOCX contains too many embedded images.")

        try:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "error",
                    Image.DecompressionBombWarning,
                )

                with Image.open(BytesIO(image_bytes)) as image:

                    width, height = image.size
                    pixels = width * height

                    if (
                        width <= 0
                        or height <= 0
                        or pixels > MAX_DOCX_IMAGE_PIXELS
                    ):
                        raise ValueError(
                            "Embedded image exceeds the pixel limit."
                        )

                    total_image_pixels += pixels

                    if total_image_pixels > MAX_TOTAL_DOCX_IMAGE_PIXELS:
                        raise ValueError(
                            "DOCX exceeds the total image pixel limit."
                        )

                    if getattr(image, "n_frames", 1) != 1:
                        continue

                    image.load()

                    grayscale = image.convert("L")

                    try:

                        check_docx_deadline(start_time)

                        text = pytesseract.image_to_string(
                            grayscale,
                            lang="eng",
                            timeout=OCR_TIMEOUT_SECONDS,
                        ).strip()

                        if text:
                            extracted_text.append(text)

                    finally:
                        grayscale.close()

        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            raise ValueError(
                "DOCX contains a potentially dangerous embedded image."
            ) from None

        except UnidentifiedImageError:
            continue

        except RuntimeError:
            raise ValueError(
                "Embedded image processing failed."
            ) from None

    return extracted_text


def load_docx(file_path):

    start_time = time.monotonic()

    validate_docx_archive(file_path)

    check_docx_deadline(start_time)

    try:
        doc = Document(file_path)

    except (
        zipfile.BadZipFile,
        KeyError,
        ValueError,
        etree.XMLSyntaxError,
        RuntimeError,
        zlib.error,
        EOFError,
        NotImplementedError,
    ):
        raise ValueError(
            "DOCX document could not be parsed safely."
        ) from None

    extracted_text = []
    total_chars = 0

    def add_text(value):

        nonlocal total_chars

        value = value.strip()

        if not value:
            return

        separator_size = 1 if extracted_text else 0
        new_size = len(value) + separator_size

        if total_chars + new_size > MAX_DOCX_EXTRACTED_CHARS:
            raise ValueError(
                "DOCX extracted text exceeds the character limit."
            )

        extracted_text.append(value)
        total_chars += new_size

    try:

        for paragraph in doc.paragraphs:

            check_docx_deadline(start_time)
            add_text(paragraph.text)

        for table in doc.tables:

            check_docx_deadline(start_time)

            for row in table.rows:

                check_docx_deadline(start_time)

                cells = []

                for cell in row.cells:

                    cell_text = cell.text.strip()

                    if cell_text:
                        cells.append(cell_text)

                if cells:
                    add_text(" | ".join(cells))

        image_text = extract_docx_images(
            doc,
            start_time,
        )

        for text in image_text:
            add_text(text)

        check_docx_deadline(start_time)

        return "\n".join(extracted_text)

    finally:
        # python-docx does not expose a public close method.
        # Its underlying package is released when references are cleared.
        del doc


# =========================================================
# 9. TXT EXTRACTION
# =========================================================

MAX_TXT_FILE_SIZE = 5 * 1024 * 1024
MAX_TXT_EXTRACTED_CHARS = 1_000_000
MAX_TXT_PROCESSING_SECONDS = 10
TXT_CHUNK_SIZE = 64 * 1024


def load_txt(file_path):

    file_path = Path(file_path)
    start_time = time.monotonic()

    if file_path.suffix.lower() != ".txt":
        raise ValueError("Invalid file extension. Expected TXT.")

    if not file_path.is_file():
        raise ValueError("TXT file is not a regular file.")

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise ValueError("TXT file is empty.")

    if file_size > MAX_TXT_FILE_SIZE:
        raise ValueError("TXT file exceeds the maximum allowed size.")

    extracted_text = []
    total_chars = 0

    with open(file_path, "rb") as file:

        prefix = file.read(3)
        has_bom = prefix == codecs.BOM_UTF8

        encoding = "utf-8-sig" if has_bom else "utf-8"

        file.seek(0)

        try:
            decoder = codecs.getincrementaldecoder(encoding)(
                errors="strict"
            )

            while True:

                if (
                    time.monotonic() - start_time
                    > MAX_TXT_PROCESSING_SECONDS
                ):
                    raise TimeoutError("TXT processing timed out.")

                chunk = file.read(TXT_CHUNK_SIZE)

                if not chunk:
                    break

                decoded = decoder.decode(chunk, final=False)

                if "\x00" in decoded:
                    raise ValueError(
                        "TXT contains null bytes and may be binary."
                    )

                total_chars += len(decoded)

                if total_chars > MAX_TXT_EXTRACTED_CHARS:
                    raise ValueError(
                        "TXT extracted content exceeds the character limit."
                    )

                extracted_text.append(decoded)

            final_chunk = decoder.decode(b"", final=True)

            if "\x00" in final_chunk:
                raise ValueError("TXT contains null bytes.")

            total_chars += len(final_chunk)

            if total_chars > MAX_TXT_EXTRACTED_CHARS:
                raise ValueError(
                    "TXT extracted content exceeds the character limit."
                )

            extracted_text.append(final_chunk)

        except UnicodeDecodeError:

            file.seek(3 if has_bom else 0)

            extracted_text = []
            total_chars = 0

            decoder = codecs.getincrementaldecoder("cp1252")(
                errors="strict"
            )

            while True:

                if (
                    time.monotonic() - start_time
                    > MAX_TXT_PROCESSING_SECONDS
                ):
                    raise TimeoutError("TXT processing timed out.")

                chunk = file.read(TXT_CHUNK_SIZE)

                if not chunk:
                    break

                decoded = decoder.decode(chunk, final=False)

                if "\x00" in decoded:
                    raise ValueError(
                        "TXT contains null bytes and may be binary."
                    )

                total_chars += len(decoded)

                if total_chars > MAX_TXT_EXTRACTED_CHARS:
                    raise ValueError(
                        "TXT extracted content exceeds the character limit."
                    )

                extracted_text.append(decoded)

            final_chunk = decoder.decode(b"", final=True)

            if "\x00" in final_chunk:
                raise ValueError("TXT contains null bytes.")

            total_chars += len(final_chunk)

            if total_chars > MAX_TXT_EXTRACTED_CHARS:
                raise ValueError(
                    "TXT extracted content exceeds the character limit."
                )

            extracted_text.append(final_chunk)

    text = "".join(extracted_text)

    if not text.strip():
        raise ValueError("TXT contains no readable text.")

    return text


# =========================================================
# 10. DOC EXTRACTION
# =========================================================

MAX_DOC_FILE_SIZE = 20 * 1024 * 1024
MAX_CONVERTED_DOCX_SIZE = 50 * 1024 * 1024
MAX_DOC_TEMP_SIZE = 100 * 1024 * 1024

DOC_CONVERSION_TIMEOUT = 45


def terminate_process_tree(process):

    try:

        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)

        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass

        try:
            parent.terminate()
        except psutil.Error:
            pass

        _, alive = psutil.wait_procs(
            children + [parent],
            timeout=3,
        )

        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass

    except psutil.Error:
        pass

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def get_directory_size(directory):

    total_size = 0

    for root, _, files in os.walk(directory):

        for filename in files:

            file_path = Path(root) / filename

            try:
                total_size += file_path.stat().st_size
            except OSError:
                continue

    return total_size


def load_doc(file_path):

    file_path = Path(file_path).resolve(strict=True)

    if file_path.suffix.lower() != ".doc":
        raise ValueError("Invalid file extension. Expected DOC.")

    if not file_path.is_file():
        raise ValueError("DOC file is not a regular file.")

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise ValueError("DOC file is empty.")

    if file_size > MAX_DOC_FILE_SIZE:
        raise ValueError("DOC file exceeds the maximum allowed size.")

    doc_signature = bytes.fromhex("D0CF11E0A1B11AE1")

    with open(file_path, "rb") as file:
        signature = file.read(8)

    if signature != doc_signature:
        raise ValueError("Invalid or unsupported legacy DOC file.")

    soffice_path = shutil.which("soffice")

    if soffice_path is None:
        raise RuntimeError(
            "LibreOffice executable not found. "
            "Install LibreOffice and configure its PATH."
        )

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        profile_dir = temp_dir / "lo_profile"
        profile_dir.mkdir()

        converted_file = temp_dir / f"{file_path.stem}.docx"

        environment = os.environ.copy()
        environment["HOME"] = str(temp_dir)
        environment["TMP"] = str(temp_dir)
        environment["TEMP"] = str(temp_dir)

        command = [
            soffice_path,
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--headless",
            "--nologo",
            "--nodefault",
            "--nolockcheck",
            "--norestore",
            "--convert-to",
            "docx:MS Word 2007 XML",
            "--outdir",
            str(temp_dir),
            str(file_path),
        ]

        process = None

        try:

            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                shell=False,
            )

            start_time = time.monotonic()

            while process.poll() is None:

                if (
                    time.monotonic() - start_time
                    > DOC_CONVERSION_TIMEOUT
                ):
                    terminate_process_tree(process)
                    raise TimeoutError(
                        "DOC to DOCX conversion timed out."
                    )

                if get_directory_size(temp_dir) > MAX_DOC_TEMP_SIZE:
                    terminate_process_tree(process)
                    raise ValueError(
                        "DOC conversion exceeded its temporary storage limit."
                    )

                time.sleep(0.25)

            if process.returncode != 0:
                raise RuntimeError(
                    "LibreOffice failed to convert the DOC file."
                )

            if not converted_file.is_file():
                raise RuntimeError(
                    "DOC conversion did not produce a DOCX file."
                )

            converted_size = converted_file.stat().st_size

            if converted_size == 0:
                raise ValueError("Converted DOCX file is empty.")

            if converted_size > MAX_CONVERTED_DOCX_SIZE:
                raise ValueError(
                    "Converted DOCX exceeds the maximum allowed size."
                )

            return load_docx(converted_file)

        except Exception:

            if process is not None and process.poll() is None:
                terminate_process_tree(process)

            raise


# =========================================================
# 11. RTF EXTRACTION
# =========================================================

MAX_RTF_FILE_SIZE = 10 * 1024 * 1024
MAX_RTF_EXTRACTED_CHARS = 1_000_000
MAX_RTF_PROCESSING_SECONDS = 10

MAX_RTF_NESTING_DEPTH = 256
MAX_RTF_GROUPS = 100_000
MAX_RTF_CONTROL_WORDS = 100_000


def validate_rtf_structure(content, start_time):

    if not content.lstrip().startswith(r"{\rtf"):
        raise ValueError("Invalid RTF file header.")

    depth = 0
    groups = 0
    control_words = 0

    i = 0
    length = len(content)

    while i < length:

        if i % 65536 == 0:

            if (
                time.monotonic() - start_time
                > MAX_RTF_PROCESSING_SECONDS
            ):
                raise TimeoutError("RTF validation timed out.")

        char = content[i]

        if char == "\\":

            i += 1

            if i >= length:
                raise ValueError("Malformed RTF control sequence.")

            next_char = content[i]

            if next_char in r"\\{}~_-*":
                i += 1
                continue

            if next_char == "'":

                if i + 2 >= length:
                    raise ValueError("Malformed RTF hexadecimal escape.")

                try:
                    int(content[i + 1:i + 3], 16)
                except ValueError:
                    raise ValueError("Invalid RTF hexadecimal escape.")

                i += 3
                continue

            if next_char.isalpha():

                start = i

                while i < length and content[i].isalpha():
                    i += 1

                word = content[start:i].lower()

                if i < length and content[i] == "-":
                    i += 1

                number_start = i

                while i < length and content[i].isdigit():
                    i += 1

                number = content[number_start:i]

                control_words += 1

                if control_words > MAX_RTF_CONTROL_WORDS:
                    raise ValueError("RTF contains too many control words.")

                if word == "bin":

                    if not number:
                        raise ValueError("Invalid RTF binary section.")

                    binary_size = int(number)

                    if binary_size < 0 or binary_size > length - i:
                        raise ValueError("Invalid RTF binary payload size.")

                    i += binary_size
                    continue

                if i < length and content[i] == " ":
                    i += 1

                continue

            i += 1
            continue

        if char == "{":

            depth += 1
            groups += 1

            if depth > MAX_RTF_NESTING_DEPTH:
                raise ValueError("RTF nesting depth exceeded.")

            if groups > MAX_RTF_GROUPS:
                raise ValueError("RTF contains too many groups.")

        elif char == "}":

            depth -= 1

            if depth < 0:
                raise ValueError("Malformed RTF group structure.")

        elif char == "\x00":
            raise ValueError("RTF contains unexpected null bytes.")

        i += 1

    if depth != 0:
        raise ValueError("RTF contains unclosed groups.")

    return True


def load_rtf(file_path):

    file_path = Path(file_path).resolve(strict=True)

    start_time = time.monotonic()

    if file_path.suffix.lower() != ".rtf":
        raise ValueError("Invalid file extension. Expected RTF.")

    if not file_path.is_file():
        raise ValueError("RTF file is not a regular file.")

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise ValueError("RTF file is empty.")

    if file_size > MAX_RTF_FILE_SIZE:
        raise ValueError("RTF file exceeds the maximum allowed size.")

    with open(file_path, "rb") as file:
        raw_content = file.read(MAX_RTF_FILE_SIZE + 1)

    if len(raw_content) > MAX_RTF_FILE_SIZE:
        raise ValueError("RTF file exceeds the maximum allowed size.")

    if raw_content.startswith(codecs.BOM_UTF8):
        raw_content = raw_content[3:]

    rtf_content = raw_content.decode("latin-1")

    validate_rtf_structure(rtf_content, start_time)

    if (
        time.monotonic() - start_time
        > MAX_RTF_PROCESSING_SECONDS
    ):
        raise TimeoutError("RTF processing timed out.")

    try:
        extracted_text = rtf_to_text(rtf_content)
    except Exception:
        raise ValueError(
            "Could not safely parse the RTF document."
        ) from None

    if not isinstance(extracted_text, str):
        raise ValueError("RTF parser returned an invalid result.")

    if len(extracted_text) > MAX_RTF_EXTRACTED_CHARS:
        raise ValueError(
            "RTF extracted content exceeds the character limit."
        )

    if not extracted_text.strip():
        raise ValueError("RTF contains no readable text.")

    if (
        time.monotonic() - start_time
        > MAX_RTF_PROCESSING_SECONDS
    ):
        raise TimeoutError("RTF processing exceeded the time limit.")

    return extracted_text


# =========================================================
# 12. IMAGE EXTRACTION
# =========================================================

MAX_IMAGE_FILE_SIZE = 10 * 1024 * 1024

MAX_IMAGE_WIDTH = 5000
MAX_IMAGE_HEIGHT = 5000
MAX_UPLOAD_IMAGE_PIXELS = 15_000_000

MAX_IMAGE_OCR_SECONDS = 15
MAX_IMAGE_EXTRACTED_CHARS = 100_000


def validate_image(file_path):

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    allowed_formats = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
    }

    if extension not in allowed_formats:
        raise ValueError("Unsupported image extension.")

    expected_format = allowed_formats[extension]

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(file_path) as image:

                if image.format != expected_format:
                    raise ValueError(
                        "Image extension does not match its actual format."
                    )

                width, height = image.size

                if width <= 0 or height <= 0:
                    raise ValueError("Image has invalid dimensions.")

                if (
                    width > MAX_IMAGE_WIDTH
                    or height > MAX_IMAGE_HEIGHT
                ):
                    raise ValueError(
                        "Image dimensions exceed the allowed limit."
                    )

                if width * height > MAX_UPLOAD_IMAGE_PIXELS:
                    raise ValueError(
                        "Image exceeds the maximum pixel limit."
                    )

                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(
                        "Animated or multi-frame images are unsupported."
                    )

                image.verify()

            with Image.open(file_path) as image:

                if image.format != expected_format:
                    raise ValueError("Image format validation failed.")

                image.load()

        return True

    except ValueError:
        raise

    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ValueError(
            "Image contains an excessive number of pixels."
        ) from None

    except (OSError, SyntaxError):
        raise ValueError(
            "Image is corrupted, truncated, or invalid."
        ) from None


def load_image(file_path):

    file_path = Path(file_path).resolve(strict=True)

    start_time = time.monotonic()

    if not file_path.is_file():
        raise ValueError("Image is not a regular file.")

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise ValueError("Image file is empty.")

    if file_size > MAX_IMAGE_FILE_SIZE:
        raise ValueError("Image exceeds the maximum file size.")

    validate_image(file_path)

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(file_path) as original_image:

                original_image.load()

                # Apply EXIF orientation before OCR.
                from PIL import ImageOps

                oriented_image = ImageOps.exif_transpose(original_image)

                try:

                    grayscale_image = oriented_image.convert("L")

                    try:

                        extracted_text = pytesseract.image_to_string(
                            grayscale_image,
                            lang="eng",
                            timeout=MAX_IMAGE_OCR_SECONDS,
                        )

                    finally:
                        grayscale_image.close()

                finally:

                    if oriented_image is not original_image:
                        oriented_image.close()

    except RuntimeError as e:

        if "timeout" in str(e).lower():
            raise TimeoutError(
                "Image OCR exceeded the processing time limit."
            ) from e

        raise RuntimeError("Image OCR processing failed.") from e

    except (
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ValueError(
            "Image could not be decoded safely."
        ) from None

    if not isinstance(extracted_text, str):
        raise ValueError("OCR returned an invalid result.")

    if len(extracted_text) > MAX_IMAGE_EXTRACTED_CHARS:
        raise ValueError("OCR output exceeds the character limit.")

    if not extracted_text.strip():
        raise ValueError("No readable text found in the image.")

    if (
        time.monotonic() - start_time
        > MAX_IMAGE_OCR_SECONDS + 5
    ):
        raise TimeoutError(
            "Image processing exceeded the time limit."
        )

    return extracted_text


# =========================================================
# 13. FILE DISPATCHER
# =========================================================

def process_file(file_path, file_type):

    if file_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type: {file_type}")

    if file_type == "pdf":
        extracted_text = load_pdf(file_path)

    elif file_type == "docx":
        extracted_text = load_docx(file_path)

    elif file_type == "txt":
        extracted_text = load_txt(file_path)

    elif file_type == "doc":
        extracted_text = load_doc(file_path)

    elif file_type == "rtf":
        extracted_text = load_rtf(file_path)

    elif file_type in {"jpeg", "png"}:
        extracted_text = load_image(file_path)

    else:
        raise ValueError("Unsupported file type.")

    if not isinstance(extracted_text, str):
        raise ValueError(
            "File extraction returned an invalid data type."
        )

    if not extracted_text.strip():
        raise ValueError("No text extracted from the uploaded file.")

    if len(extracted_text) > MAX_EXTRACTED_TEXT_CHARS:
        raise ValueError(
            "Extracted content exceeds the application-wide limit."
        )

    return extracted_text


# =========================================================
# 14. WORKER POOL AND RESOURCE PROTECTION
# =========================================================

WORKER_TIMEOUTS = {
    "pdf": 55,
    "docx": 75,
    "doc": 120,
    "rtf": 25,
    "txt": 15,
    "jpeg": 25,
    "png": 25,
}

MAX_WORKERS = 2
WORKER_MEMORY_LIMIT_MB = 1024
MAX_TASKS_PER_WORKER = 20

extraction_pool = None
worker_ready_queue = None

_worker_job_handle = None
_worker_memory_limit_error = None


def initialize_worker_memory_limit(ready_queue):

    global _worker_job_handle
    global _worker_memory_limit_error

    job = None

    try:

        job = win32job.CreateJobObject(None, "")

        job_info = win32job.QueryInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
        )

        job_info["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
            | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )

        job_info["JobMemoryLimit"] = (
            WORKER_MEMORY_LIMIT_MB * 1024 * 1024
        )

        win32job.SetInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
            job_info,
        )

        win32job.AssignProcessToJobObject(
            job,
            win32api.GetCurrentProcess(),
        )

        _worker_job_handle = job
        _worker_memory_limit_error = None

        print(
            f"[MEMORY] Worker {win32api.GetCurrentProcessId()} "
            f"protected with {WORKER_MEMORY_LIMIT_MB} MB limit."
        )

        ready_queue.put(
            (os.getpid(), True, None)
        )

    except Exception as e:

        if job is not None:
            try:
                job.Close()
            except Exception:
                pass

        _worker_job_handle = None

        _worker_memory_limit_error = (
            f"{type(e).__name__}: {e}"
        )

        ready_queue.put(
            (os.getpid(), False, _worker_memory_limit_error)
        )

# def initialize_worker_memory_limit(ready_queue):

#     pid = os.getpid()

#     print(
#         f"[INIT] Worker {pid} started.",
#         flush=True
#     )

#     ready_queue.put((pid, True, None))

#     print(
#         f"[INIT] Worker {pid} reported readiness.",
#         flush=True
#     )


def extraction_worker(file_path, file_type):

    if _worker_memory_limit_error is not None:
        raise RuntimeError(
            "Worker memory protection is unavailable. "
            f"Extraction blocked: {_worker_memory_limit_error}"
        )

    process = psutil.Process(os.getpid())

    start_time = time.perf_counter()

    stop_monitoring = threading.Event()

    initial_memory = 0
    peak_memory = 0

    def get_process_tree_memory():

        total_memory = 0

        try:

            total_memory += process.memory_info().rss

            for child in process.children(recursive=True):

                try:
                    total_memory += child.memory_info().rss
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                ):
                    continue

        except psutil.NoSuchProcess:
            pass

        return total_memory

    initial_memory = get_process_tree_memory()
    peak_memory = initial_memory

    def monitor_memory():

        nonlocal peak_memory

        while not stop_monitoring.wait(0.1):

            current_memory = get_process_tree_memory()

            peak_memory = max(
                peak_memory,
                current_memory,
            )

    monitor_thread = threading.Thread(
        target=monitor_memory,
        daemon=True,
    )

    monitor_thread.start()

    try:

        extracted_text = process_file(
            file_path,
            file_type,
        )

    finally:

        stop_monitoring.set()
        monitor_thread.join()

        final_memory = get_process_tree_memory()

        peak_memory = max(
            peak_memory,
            final_memory,
        )

    extraction_time = time.perf_counter() - start_time

    return (
        extracted_text,
        extraction_time,
        initial_memory / (1024 ** 2),
        peak_memory / (1024 ** 2),
        os.getpid(),
    )


def initialize_extraction_pool():

    global extraction_pool
    global worker_ready_queue

    if extraction_pool is not None:
        return extraction_pool

    start_time = time.perf_counter()

    context = mp.get_context("spawn")

    worker_ready_queue = context.Queue()

    extraction_pool = context.Pool(
        processes=MAX_WORKERS,
        maxtasksperchild=MAX_TASKS_PER_WORKER,
        initializer=initialize_worker_memory_limit,
        initargs=(worker_ready_queue,),
    )

    pool_creation_time = time.perf_counter() - start_time

    print(
        f"\n[TIMING] Pool creation: {pool_creation_time:.4f}s"
    )

    # Wait for each worker's initializer to report its status.
    start_time = time.perf_counter()

    ready_workers = set()
    initialization_errors = []

    readiness_timeout = 30
    deadline = time.monotonic() + readiness_timeout

    while len(ready_workers) < MAX_WORKERS:

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            print("\n[DEBUG] Worker initialization timed out.")

            for worker in extraction_pool._pool:
                print(
                    f"PID: {worker.pid} | "
                    f"Alive: {worker.is_alive()} | "
                    f"Exit code: {worker.exitcode}"
                )

            print("[DEBUG] Ready workers:", ready_workers)
            print("[DEBUG] Initialization errors:", initialization_errors)
            extraction_pool.terminate()
            extraction_pool.join()
            extraction_pool = None

            raise TimeoutError(
                "Worker processes did not initialize in time."
            )

        try:
            pid, success, error = worker_ready_queue.get(
                timeout=min(1, remaining)
            )
        except Exception:
            continue

        ready_workers.add(pid)

        if not success:
            initialization_errors.append(
                f"Worker {pid}: {error}"
            )

    worker_ready_time = time.perf_counter() - start_time

    print(
        f"[TIMING] Worker readiness: {worker_ready_time:.4f}s"
    )

    if initialization_errors:

        extraction_pool.terminate()
        extraction_pool.join()
        extraction_pool = None

        raise RuntimeError(
            "Worker security initialization failed: "
            + "; ".join(initialization_errors)
        )

    return extraction_pool


def run_extraction_worker(file_path, file_type):

    global extraction_pool

    pool = initialize_extraction_pool()

    timeout = WORKER_TIMEOUTS.get(file_type, 30)

    start_time = time.perf_counter()

    try:

        result = pool.apply_async(
            extraction_worker,
            args=(str(file_path), file_type),
        )

        (
            extracted_text,
            extraction_time,
            initial_memory,
            peak_memory,
            worker_pid,
        ) = result.get(timeout=timeout)

        print(f"[MEMORY] Worker PID: {worker_pid}")

        print(
            f"[MEMORY] Initial usage: {initial_memory:.2f} MB"
        )

        print(
            f"[MEMORY] Peak usage: {peak_memory:.2f} MB"
        )

        print(
            f"[TIMING] Actual extraction: {extraction_time:.4f}s"
        )

        print(
            f"[TIMING] Total task time: "
            f"{time.perf_counter() - start_time:.4f}s"
        )

        return extracted_text

    except mp.TimeoutError:

        print(
            f"\n[TIMING] {file_type.upper()} extraction timed out "
            f"after {timeout} seconds."
        )

        pool.terminate()
        pool.join()

        extraction_pool = None

        raise TimeoutError(
            f"{file_type.upper()} extraction exceeded "
            f"the {timeout}-second worker timeout. "
            "The pool will be recreated on its next use."
        )

    except Exception:
        raise


def shutdown_extraction_pool():

    global extraction_pool
    global worker_ready_queue

    if extraction_pool is None:
        return

    pool = extraction_pool

    extraction_pool = None

    try:

        pool.close()
        pool.join()

        print("\nExtraction worker pool shut down.")

    except Exception:

        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass

        raise

    finally:

        if worker_ready_queue is not None:

            try:
                worker_ready_queue.close()
                worker_ready_queue.join_thread()
            except Exception:
                pass

            worker_ready_queue = None


# =========================================================
# 15. MAIN FILE LOADER
# =========================================================


import unicodedata




def validate_extracted_text(extracted_text):

    if not isinstance(extracted_text, str):
        raise ValueError(
            "Extraction worker returned an invalid result."
        )

    if not extracted_text.strip():
        raise ValueError(
            "No text extracted from the uploaded file."
        )

    if len(extracted_text) > MAX_EXTRACTED_TEXT_CHARS:
        raise ValueError(
            "Extracted content exceeds the application-wide limit."
        )

    # Reject strings that cannot be encoded as valid UTF-8.
    try:
        extracted_text.encode("utf-8", errors="strict")

    except UnicodeEncodeError as exc:
        raise ValueError(
            "Extracted text contains invalid Unicode characters."
        ) from exc

    # Reject null bytes.
    if "\x00" in extracted_text:
        raise ValueError(
            "Extracted text contains null bytes."
        )

    # Allow common document control characters.
    allowed_controls = {
        "\t",
        "\n",
        "\r",
        "\f",
    }

    # Reject unexpected Unicode control characters.
    for character in extracted_text:

        if (
            unicodedata.category(character) == "Cc"
            and character not in allowed_controls
        ):
            raise ValueError(
                "Extracted text contains unexpected control characters."
            )

    return extracted_text

from resume_utils import calculate_experience




def load_file(file_path):

    file_path = Path(file_path).resolve(strict=True)

    if not file_path.is_file():
        raise ValueError("Uploaded file is not a regular file.")

    # ==================================================
    # 1. BRONZE
    # ==================================================

    validation = validate_resume_file(file_path)

    file_type = validation["file_type"]

    if file_type not in SUPPORTED_FILE_TYPES:
        raise ValueError("Unsupported uploaded file type.")

    fingerprint = generate_fingerprint(file_path)

    existing_bronze = get_bronze_by_fingerprint(fingerprint)

    # ==================================================
    # 2. PRIVATE STAGING + SECURITY
    # ==================================================

    with tempfile.TemporaryDirectory(prefix="resumeiq_") as staging_directory:

        staged_file = Path(staging_directory) / file_path.name

        shutil.copy2(file_path, staged_file)

        if generate_fingerprint(staged_file) != fingerprint:
            raise ValueError("File changed while being staged.")

        staged_validation = validate_resume_file(staged_file)

        if staged_validation["file_type"] != file_type:
            raise ValueError("Staged file type differs from original.")

        scan_with_clamav(staged_file)

        print("Antivirus scan passed.")

        # ==================================================
        # 3. BRONZE CACHE OR EXTRACTION
        # ==================================================

        if existing_bronze:

            if existing_bronze["file_type"] != file_type:
                raise ValueError("Stored Bronze file type mismatch.")

            bronze_id = existing_bronze["id"]

            extracted_text = existing_bronze["extracted_text"]

            print(f"Bronze cache HIT: {bronze_id}")

        else:

            print("Bronze cache MISS.")

            extracted_text = run_extraction_worker(
                staged_file,
                file_type
            )

            extracted_text = validate_extracted_text(extracted_text)

            if generate_fingerprint(file_path) != fingerprint:
                raise ValueError("Original file changed during extraction.")

            bronze_record = save_bronze(
                fingerprint=fingerprint,
                filename=file_path.name,
                file_type=file_type,
                extracted_text=extracted_text,
                mime_type=validation["mime_type"],
                source_file_size=validation["size_bytes"],
                extraction_version="v1"
            )

            bronze_id = bronze_record["id"]

            extracted_text = bronze_record["extracted_text"]

            print(f"Bronze saved: {bronze_id}")

    # ==================================================
    # 4. SILVER CACHE OR PROCESSING
    # ==================================================

    silver_result = process_silver(bronze_id)

    silver_id = silver_result["id"]

    print(f"Silver ID: {silver_id}")

    # ==================================================
    # 5. GOLD CACHE OR LLM EXTRACTION
    # ==================================================

    resume_json = process_gold(silver_id)

    print("Gold processing completed.")

    # ==================================================
    # 6. EXPERIENCE CALCULATION
    # ==================================================

    experience_data = calculate_experience(
        resume_json["experience"]
    )

    print("Experience:", experience_data)

    # ==================================================
    # 7. FINAL RESULT
    # ==================================================

    return {
        "resume_json": resume_json,
        "fingerprint": fingerprint,
        "bronze_id": bronze_id,
        "silver_id": silver_id,
        "experience_data": experience_data
    }


# =========================================================
# 17. MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":

    mp.freeze_support()

    # try:

    #     initialize_database()
    #     initialize_extraction_pool()

    #     test_resume_files(
    #         folder_path="dummy_resumes",
    #         filenames=[
    #             # "average_01.pdf",
    #             # "average_02.pdf",
    #             # "strong_01.pdf",
    #             # "strong_02.pdf",
    #             "strong_07.pdf"
    #         ],
    #     )

    # finally:

    #     shutdown_extraction_pool()
