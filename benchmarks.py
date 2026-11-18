
import csv
import json
import sqlite3
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# =========================================================
# 1. CONFIGURATION
# =========================================================

DB_PATH = "storage/benchmarks.db"
CHROMA_PATH = "storage/chroma_db"
CHROMA_COLLECTION = "resume_skilks"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Dense retrieval weights
DENSE_WEIGHTS = {
    "technical_skills": 0.30,
    "projects": 0.30,
    "experience": 0.25,
    "professional_summary": 0.10,
    "certifications": 0.05
}

# Sparse retrieval weights
SPARSE_WEIGHTS = {
    "skills": 45.0,
    "projects": 30.0,
    "experience": 20.0,
    "certifications": 5.0
}

REQUIRED_COLUMNS = [
    "resume_id",
    "file",
    "candidate_name",
    "candidate_level",
    "role_name",
    "human_score",
    "quality_label",
    "synthetic_data",
    "resume_json",
    "category_evaluations",
    "score_reason",
    "rubric_version",
    "schema_version",
    "evaluation_status"
]


# =========================================================
# 2. INITIALIZE SQLITE DATABASE
# =========================================================

def initialize_database():

    Path("storage").mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.executescript("""

        CREATE TABLE IF NOT EXISTS benchmark_resumes (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            resume_id TEXT NOT NULL UNIQUE,
            file TEXT NOT NULL,
            candidate_name TEXT NOT NULL,

            candidate_level TEXT NOT NULL
                CHECK (
                    candidate_level IN (
                        'Fresher',
                        'Junior',
                        'Experienced'
                    )
                ),

            role_name TEXT NOT NULL,

            human_score REAL NOT NULL
                CHECK (human_score >= 0 AND human_score <= 10),

            quality_label TEXT NOT NULL
                CHECK (
                    quality_label IN (
                        'Strong',
                        'Average',
                        'Weak'
                    )
                ),

            synthetic_data INTEGER NOT NULL,

            resume_json TEXT NOT NULL,
            category_evaluations TEXT NOT NULL,
            score_reason TEXT NOT NULL,

            rubric_version TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            evaluation_status TEXT NOT NULL
        );


        CREATE TABLE IF NOT EXISTS role_guides (

            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT NOT NULL UNIQUE,
            summary TEXT NOT NULL
        );


        CREATE VIRTUAL TABLE IF NOT EXISTS benchmark_fts
        USING fts5(

            resume_id UNINDEXED,
            role_name UNINDEXED,
            candidate_level UNINDEXED,

            skills,
            projects,
            experience,
            certifications
        );

    """)

    connection.commit()
    connection.close()

    print("SQLite database initialized.")


# =========================================================
# 3. INITIALIZE CHROMA
# =========================================================

# =========================================================
# 3. LAZY INITIALIZE CHROMA
# =========================================================

embedding_model = None
vector_store = None


def get_vector_store():

    global embedding_model
    global vector_store

    if vector_store is None:

        print("[DENSE] Initializing embedding model...")

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory=CHROMA_PATH,
            collection_name=CHROMA_COLLECTION
        )

        print("[DENSE] Chroma initialized.")

    return vector_store


# =========================================================
# 4. SQLITE BENCHMARK OPERATIONS
# =========================================================

def check_benchmark(resume_id):

    connection = sqlite3.connect(DB_PATH)

    result = connection.execute("""
        SELECT resume_id
        FROM benchmark_resumes
        WHERE resume_id = ?
    """, (resume_id,)).fetchone()

    connection.close()

    return result is not None


def fetch_benchmark(resume_id):

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    result = connection.execute("""
        SELECT *
        FROM benchmark_resumes
        WHERE resume_id = ?
    """, (resume_id,)).fetchone()

    connection.close()

    return dict(result) if result else None


def save_benchmark(record):

    connection = sqlite3.connect(DB_PATH)

    try:

        connection.execute("""

            INSERT OR IGNORE INTO benchmark_resumes (

                resume_id,
                file,
                candidate_name,
                candidate_level,
                role_name,
                human_score,
                quality_label,
                synthetic_data,
                resume_json,
                category_evaluations,
                score_reason,
                rubric_version,
                schema_version,
                evaluation_status

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            record["resume_id"],
            record["file"],
            record["candidate_name"],
            record["candidate_level"],
            record["role_name"],
            record["human_score"],
            record["quality_label"],
            int(record["synthetic_data"]),
            record["resume_json"],
            record["category_evaluations"],
            record["score_reason"],
            record["rubric_version"],
            record["schema_version"],
            record["evaluation_status"]

        ))

        connection.commit()

    finally:

        connection.close()


# =========================================================
# 5. BUILD SEARCHABLE RESUME SECTIONS
# =========================================================

def build_resume_fields(resume_json):

    skills = resume_json.get("skills") or {}

    # Technical skills only.
    # Soft skills are deliberately excluded.

    skill_categories = [
        "programming_languages",
        "frameworks_and_libraries",
        "databases",
        "tools_and_platforms",
        "algorithms_and_models",
        "methodologies",
        "domain_knowledge"
    ]

    skill_parts = []

    for category in skill_categories:

        values = skills.get(category) or []

        if values:

            skill_parts.append(
                f"{category}: {' '.join(map(str, values))}"
            )

    skills_text = " ".join(skill_parts)


    # Projects

    project_parts = []

    for project in resume_json.get("projects") or []:

        project_parts.append(

            f"Project: {project.get('project_name') or ''}. "

            f"Description: {project.get('description') or ''}. "

            f"Technologies: "
            f"{' '.join(project.get('technologies') or [])}. "

            f"Features: "
            f"{' '.join(project.get('key_features') or [])}. "

            f"Methods: "
            f"{' '.join(project.get('algorithms_and_methods') or [])}."

        )

    projects_text = " ".join(project_parts)


    # Professional experience

    experience_parts = []

    for exp in resume_json.get("experience") or []:

        experience_parts.append(

            f"Job title: {exp.get('job_title') or ''}. "

            f"Employment type: "
            f"{exp.get('employment_type') or ''}. "

            f"Responsibilities: "
            f"{' '.join(exp.get('responsibilities') or [])}. "

            f"Technologies: "
            f"{' '.join(exp.get('technologies_and_tools') or [])}. "

            f"Methodologies: "
            f"{' '.join(exp.get('methodologies') or [])}. "

            f"Achievements: "
            f"{' '.join(exp.get('achievements') or [])}. "

            f"Measurable outcomes: "
            f"{' '.join(exp.get('measurable_outcomes') or [])}."

        )

    experience_text = " ".join(experience_parts)


    # Certifications

    certification_parts = []

    for certification in resume_json.get("certifications") or []:

        if isinstance(certification, str):

            certification_parts.append(certification)

        elif isinstance(certification, dict):

            certification_parts.append(
                json.dumps(certification, ensure_ascii=False)
            )

    certifications_text = " ".join(certification_parts)


    # Professional summary

    summary_text = (
        resume_json.get("professional_summary") or ""
    )


    return {

        "technical_skills": skills_text,
        "projects": projects_text,
        "experience": experience_text,
        "professional_summary": summary_text,
        "certifications": certifications_text

    }


# =========================================================
# 6. BUILD DENSE DOCUMENTS FOR CHROMA
# =========================================================

def build_dense_documents(record):

    resume_json = json.loads(record["resume_json"])

    fields = build_resume_fields(resume_json)

    sections = {

        "technical_skills": fields["technical_skills"],
        "projects": fields["projects"],
        "experience": fields["experience"],
        "professional_summary": fields["professional_summary"],
        "certifications": fields["certifications"]

    }

    documents = []

    for section, content in sections.items():

        # Don't create empty embeddings.

        if not content.strip():
            continue

        metadata = {

            "resume_id": record["resume_id"],
            "section": section,

            "candidate_name": record["candidate_name"],
            "candidate_level": record["candidate_level"],
            "role_name": record["role_name"],

            "human_score": float(record["human_score"]),
            "quality_label": record["quality_label"],

            "synthetic_data": bool(record["synthetic_data"]),

            "rubric_version": record["rubric_version"],
            "schema_version": record["schema_version"],
            "evaluation_status": record["evaluation_status"],

            "section_weight": DENSE_WEIGHTS[section]

        }

        documents.append(

            Document(
                page_content=content,
                metadata=metadata
            )

        )

    return documents


# =========================================================
# 7. CHROMA DENSE EMBEDDING OPERATIONS
# =========================================================

def get_embedding_ids(resume_id):

    response = get_vector_store().get(
        where={"resume_id": resume_id},
        include=[]
    )

    return response.get("ids", [])


def embedding_exists(resume_id):

    return len(get_embedding_ids(resume_id)) > 0


def add_embeddings(record):

    documents = build_dense_documents(record)

    existing_ids = set(get_embedding_ids(record["resume_id"]))

    new_documents = []
    new_ids = []

    for document in documents:

        section = document.metadata["section"]

        document_id = (
            f"{record['resume_id']}__{section}"
        )

        if document_id in existing_ids:

            print(f"Dense section already exists: {section}")
            continue

        new_documents.append(document)
        new_ids.append(document_id)

    if not new_documents:

        print("All dense sections already exist.")
        return

    get_vector_store().add_documents(

        documents=new_documents,
        ids=new_ids

    )

    print(
        f"Created {len(new_documents)} dense sections "
        f"for {record['resume_id']}."
    )


# =========================================================
# 8. FTS5 SPARSE OPERATIONS
# =========================================================

def sparse_embedding_exists(resume_id):

    connection = sqlite3.connect(DB_PATH)

    result = connection.execute("""

        SELECT resume_id
        FROM benchmark_fts
        WHERE resume_id = ?
        LIMIT 1

    """, (resume_id,)).fetchone()

    connection.close()

    return result is not None


def add_sparse_embedding(record):

    resume_json = json.loads(record["resume_json"])

    fields = build_resume_fields(resume_json)

    connection = sqlite3.connect(DB_PATH)

    try:

        connection.execute("""

            INSERT INTO benchmark_fts (

                resume_id,
                role_name,
                candidate_level,
                skills,
                projects,
                experience,
                certifications

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

        """, (

            record["resume_id"],
            record["role_name"],
            record["candidate_level"],

            fields["technical_skills"],
            fields["projects"],
            fields["experience"],
            fields["certifications"]

        ))

        connection.commit()

    finally:

        connection.close()

    print(f"Sparse record created: {record['resume_id']}")


# =========================================================
# 9. VALIDATE CSV RECORDS
# =========================================================

def parse_boolean(value):

    value = str(value).strip().lower()

    if value in ("true", "1", "yes"):
        return True

    if value in ("false", "0", "no"):
        return False

    raise ValueError(
        f"Invalid synthetic_data value: {value}"
    )


def validate_record(row):

    missing = [

        column
        for column in REQUIRED_COLUMNS
        if row.get(column) is None or not row[column].strip()

    ]

    if missing:

        raise ValueError(
            f"Missing required fields: {missing}"
        )

    if row["candidate_level"] not in (
        "Fresher",
        "Junior",
        "Experienced"
    ):

        raise ValueError("Invalid candidate_level.")

    if row["quality_label"] not in (
        "Strong",
        "Average",
        "Weak"
    ):

        raise ValueError("Invalid quality_label.")

    human_score = float(row["human_score"])

    if not 0 <= human_score <= 10:

        raise ValueError(
            "human_score must be between 0 and 10."
        )

    resume_json = json.loads(row["resume_json"])

    category_evaluations = json.loads(
        row["category_evaluations"]
    )

    if not isinstance(resume_json, dict):

        raise ValueError(
            "resume_json must be a JSON object."
        )

    if not isinstance(category_evaluations, dict):

        raise ValueError(
            "category_evaluations must be a JSON object."
        )

    return {

        "resume_id": row["resume_id"].strip(),
        "file": row["file"].strip(),
        "candidate_name": row["candidate_name"].strip(),
        "candidate_level": row["candidate_level"].strip(),
        "role_name": row["role_name"].strip(),

        "human_score": human_score,
        "quality_label": row["quality_label"].strip(),

        "synthetic_data": parse_boolean(
            row["synthetic_data"]
        ),

        "resume_json": json.dumps(
            resume_json,
            ensure_ascii=False
        ),

        "category_evaluations": json.dumps(
            category_evaluations,
            ensure_ascii=False
        ),

        "score_reason": row["score_reason"].strip(),

        "rubric_version": row["rubric_version"].strip(),
        "schema_version": row["schema_version"].strip(),
        "evaluation_status": row["evaluation_status"].strip()

    }


# =========================================================
# 10. ADD ONE BENCHMARK TO ALL STORAGE LAYERS
# =========================================================

def add_benchmark(record):

    resume_id = record["resume_id"]

    print(f"\nProcessing: {resume_id}")


    # ---------------------------------------------
    # SQLITE
    # ---------------------------------------------

    if check_benchmark(resume_id):

        print("Benchmark already exists in SQLite.")

        stored_record = fetch_benchmark(resume_id)

    else:

        print("Benchmark missing in SQLite. Adding it.")

        save_benchmark(record)

        stored_record = fetch_benchmark(resume_id)

        print("Benchmark saved in SQLite.")


    # Always use the existing SQLite record as
    # the source for rebuilding missing indexes.


    # ---------------------------------------------
    # CHROMA
    # ---------------------------------------------

    if embedding_exists(resume_id):

        print("Dense embedding already exists in Chroma.")

        # add_embeddings checks individual sections too.
        # This also repairs partially missing sections.

    else:

        print("Dense embedding missing in Chroma.")

    add_embeddings(stored_record)


    # ---------------------------------------------
    # FTS5
    # ---------------------------------------------

    if sparse_embedding_exists(resume_id):

        print("Sparse record already exists in FTS5.")

    else:

        print("Sparse record missing in FTS5.")

        add_sparse_embedding(stored_record)


    print(f"Completed: {resume_id}")


# =========================================================
# 11. INGEST MULTIPLE BENCHMARKS FROM CSV
# =========================================================

def add_multi_benchmarks(csv_path):

    csv_path = Path(csv_path)

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Benchmark CSV not found: {csv_path}"
        )

    seen_ids = set()

    with open(
        csv_path,
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:

            raise ValueError("CSV has no header.")

        missing_columns = (
            set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        )

        if missing_columns:

            raise ValueError(
                f"CSV missing columns: {missing_columns}"
            )

        for row_number, row in enumerate(reader, start=2):

            try:

                record = validate_record(row)

                resume_id = record["resume_id"]

                if resume_id in seen_ids:

                    raise ValueError(
                        f"Duplicate resume_id in CSV: {resume_id}"
                    )

                seen_ids.add(resume_id)

                add_benchmark(record)

            except Exception as error:

                print(
                    f"FAILED at CSV row {row_number}: "
                    f"{type(error).__name__}: {error}"
                )



# =========================================================
# 12. ROLE GUIDE OPERATIONS
# =========================================================



def get_role_guide(role_name):

    connection = sqlite3.connect(DB_PATH)

    result = connection.execute("""

        SELECT summary
        FROM role_guides
        WHERE role_name = ?

    """, (role_name,)).fetchone()

    connection.close()

    return result[0] if result else None


def add_role_guide(role_name, role_summary):

    connection = sqlite3.connect(DB_PATH)

    try:

        connection.execute("""

            INSERT OR IGNORE INTO role_guides (
                role_name,
                summary
            )

            VALUES (?, ?)

        """, (role_name, role_summary))

        connection.commit()

    finally:

        connection.close()

    print(f"Role guide checked: {role_name}")


def update_role_guide(role_name, new_role_guide):

    connection = sqlite3.connect(DB_PATH)

    try:

        cursor = connection.execute("""

            UPDATE role_guides
            SET summary = ?
            WHERE role_name = ?

        """, (new_role_guide, role_name))

        connection.commit()

        updated = cursor.rowcount

    finally:

        connection.close()

    if updated:
        print(f"Role guide updated: {role_name}")
    else:
        print(f"Role guide not found: {role_name}")

    return updated > 0


# =========================================================
# 15. MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":

    initialize_database()

    add_multi_benchmarks(
        csv_path="benchmark.csv"
    )
