
import json
import re
import sqlite3

from extraction import load_file

from benchmarks import (
    get_vector_store,
    DENSE_WEIGHTS,
    SPARSE_WEIGHTS
)


DB_PATH = "storage/benchmarks.db"


# =========================================================
# 1. BUILD DENSE RETRIEVAL QUERY
# =========================================================

def build_dense_query(resume_data):

    query_parts = []

    # Technical skills

    skills = resume_data.get("skills", {})

    technical_categories = [
        "programming_languages",
        "frameworks_and_libraries",
        "databases",
        "tools_and_platforms",
        "algorithms_and_models",
        "methodologies",
        "domain_knowledge"
    ]

    for category in technical_categories:

        values = skills.get(category, [])

        if isinstance(values, list):
            query_parts.extend(values)

    # Projects

    for project in resume_data.get("projects", []):

        for field in [
            "project_name",
            "description",
            "technologies",
            "key_features",
            "algorithms_and_methods"
        ]:

            value = project.get(field)

            if isinstance(value, list):
                query_parts.extend(value)

            elif isinstance(value, str):
                query_parts.append(value)

    # Professional experience

    for experience in resume_data.get("experience", []):

        for field in [
            "job_title",
            "responsibilities",
            "technologies_and_tools",
            "methodologies",
            "achievements",
            "measurable_outcomes"
        ]:

            value = experience.get(field)

            if isinstance(value, list):
                query_parts.extend(value)

            elif isinstance(value, str):
                query_parts.append(value)

    # Professional summary

    summary = resume_data.get("professional_summary")

    if isinstance(summary, str):
        query_parts.append(summary)

    # Certifications

    for certification in resume_data.get("certifications", []):

        if isinstance(certification, str):
            query_parts.append(certification)

        elif isinstance(certification, dict):

            query_parts.extend(
                str(value)
                for value in certification.values()
                if value
            )

    # Remove empty values

    query_parts = [
        str(value).strip()
        for value in query_parts
        if value and str(value).strip()
    ]

    return " ".join(query_parts)


# =========================================================
# 2. BUILD SPARSE BM25 QUERY
# =========================================================

def build_sparse_query(resume_data):

    query = {
        "skills": [],
        "projects": [],
        "experience": [],
        "certifications": []
    }

    # Technical skills

    skills = resume_data.get("skills", {})

    technical_categories = [
        "programming_languages",
        "frameworks_and_libraries",
        "databases",
        "tools_and_platforms",
        "algorithms_and_models",
        "methodologies",
        "domain_knowledge"
    ]

    for category in technical_categories:

        values = skills.get(category, [])

        if isinstance(values, list):
            query["skills"].extend(values)

    # Projects

    for project in resume_data.get("projects", []):

        for field in [
            "project_name",
            "technologies",
            "description",
            "key_features",
            "algorithms_and_methods"
        ]:

            value = project.get(field)

            if isinstance(value, list):
                query["projects"].extend(value)

            elif isinstance(value, str):
                query["projects"].append(value)

    # Professional experience

    for experience in resume_data.get("experience", []):

        for field in [
            "job_title",
            "responsibilities",
            "technologies_and_tools",
            "methodologies",
            "achievements",
            "measurable_outcomes"
        ]:

            value = experience.get(field)

            if isinstance(value, list):
                query["experience"].extend(value)

            elif isinstance(value, str):
                query["experience"].append(value)

    # Certifications

    for certification in resume_data.get("certifications", []):

        if isinstance(certification, str):
            query["certifications"].append(certification)

        elif isinstance(certification, dict):

            query["certifications"].extend(
                str(value)
                for value in certification.values()
                if value
            )

    # Remove empty values and duplicates

    for category in query:

        query[category] = list(dict.fromkeys(
            str(value).strip()
            for value in query[category]
            if value and str(value).strip()
        ))

    return query


# =========================================================
# 3. BUILD FTS5 QUERY
# =========================================================

def build_fts_query(query):

    query_parts = []

    column_map = {
        "skills": "skills",
        "projects": "projects",
        "experience": "experience",
        "certifications": "certifications"
    }

    for category, column in column_map.items():

        terms = query.get(category, [])

        words = list(dict.fromkeys(
            word.lower()
            for term in terms
            for word in re.findall(r"\w+", term.lower())
            if len(word) > 1
        ))

        if words:

            category_query = " OR ".join(
                f'"{word}"'
                for word in words
            )

            query_parts.append(
                f"{column} : ({category_query})"
            )

    if not query_parts:
        return None

    return " OR ".join(query_parts)


# =========================================================
# 4. DENSE RETRIEVAL - CHROMA
# =========================================================

def dense_retrieve(
    query,
    role_name,
    candidate_level,
    k=10,
    fetch_k=20
):

    scores = {}

    for section, weight in DENSE_WEIGHTS.items():

        # Strict role and experience-level filtering

        search_filter = {
            "$and": [
                {"section": section},
                {"role_name": role_name},
                {"candidate_level": candidate_level}
            ]
        }

        results = get_vector_store().similarity_search_with_relevance_scores(

            query,
            k=fetch_k,
            filter=search_filter

        )

        for document, similarity in results:

            resume_id = document.metadata["resume_id"]

            weighted_score = max(0.0, similarity) * weight

            scores[resume_id] = (
                scores.get(resume_id, 0.0) + weighted_score
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked[:k]


# =========================================================
# 5. SPARSE RETRIEVAL - SQLITE FTS5 BM25
# =========================================================

def sparse_retrieve(
    query,
    role_name,
    candidate_level,
    k=10
):

    connection = sqlite3.connect(DB_PATH)

    try:

        sql = """

            SELECT
                resume_id,

                bm25(
                    benchmark_fts,
                    0.0,
                    0.0,
                    0.0,
                    ?,
                    ?,
                    ?,
                    ?
                ) AS score

            FROM benchmark_fts

            WHERE benchmark_fts MATCH ?

            AND role_name = ?

            AND candidate_level = ?

            ORDER BY score ASC

            LIMIT ?

        """

        params = [

            SPARSE_WEIGHTS["skills"],
            SPARSE_WEIGHTS["projects"],
            SPARSE_WEIGHTS["experience"],
            SPARSE_WEIGHTS["certifications"],

            query,
            role_name,
            candidate_level,
            k

        ]

        results = connection.execute(
            sql,
            params
        ).fetchall()

    finally:

        connection.close()

    return results


# =========================================================
# 6. RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    k=60,
    dense_weight=0.25,
    sparse_weight=0.75
):

    scores = {}

    # Combine dense and sparse rankings

    for results, weight in [
        (dense_results, dense_weight),
        (sparse_results, sparse_weight)
    ]:

        for rank, (resume_id, _) in enumerate(results, start=1):

            if resume_id not in scores:
                scores[resume_id] = 0.0

            scores[resume_id] += weight / (k + rank)

    # Sort by RRF score

    ranked_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results


# =========================================================
# 7. FETCH BENCHMARK RESUMES FROM SQLITE
# =========================================================

def fetch_benchmarks(ranked_results):

    retrieved_resumes = []

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    try:

        for resume_id, rrf_score in ranked_results:

            result = connection.execute("""

                SELECT
                    resume_id,
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

                FROM benchmark_resumes

                WHERE resume_id = ?

            """, (resume_id,)).fetchone()

            if result:

                retrieved_resumes.append({

                    "resume_id": result["resume_id"],

                    "candidate_name": result["candidate_name"],

                    "candidate_level": result["candidate_level"],

                    "role_name": result["role_name"],

                    "rrf_score": rrf_score,

                    "human_score": result["human_score"],

                    "quality_label": result["quality_label"],

                    "synthetic_data": result["synthetic_data"],

                    "resume": json.loads(result["resume_json"]),

                    "category_evaluations": json.loads(
                        result["category_evaluations"]
                    ) if result["category_evaluations"] else {},

                    "score_reason": result["score_reason"],

                    "rubric_version": result["rubric_version"],

                    "schema_version": result["schema_version"],

                    "evaluation_status": result["evaluation_status"]

                })

    finally:

        connection.close()

    return retrieved_resumes





# =========================================================
# 8. DISPLAY INDIVIDUAL RETRIEVAL RESULTS
# =========================================================

def display_retrieval_results(title, results):

    print(f"\n{'=' * 55}")
    print(title)
    print(f"{'=' * 55}")

    if not results:
        print("No results found.")
        return

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:

        for rank, (resume_id, score) in enumerate(results, start=1):

            result = connection.execute("""

                SELECT
                    candidate_name,
                    role_name,
                    candidate_level,
                    human_score,
                    quality_label

                FROM benchmark_resumes

                WHERE resume_id = ?

            """, (resume_id,)).fetchone()

            if result:

                print(f"\nRank: {rank}")
                print("Resume ID:", resume_id)
                print("Candidate:", result["candidate_name"])
                print("Role:", result["role_name"])
                print("Level:", result["candidate_level"])
                print("Human Score:", result["human_score"])
                print("Quality:", result["quality_label"])
                print("Retrieval Score:", round(score, 6))

    finally:

        connection.close()




# =========================================================
# 8. ROLE-SPECIFIC + EXPERIENCE-LEVEL RETRIEVER
# =========================================================

def retrieve_role_level(
    file_path,
    role_name,
    k,
    dense_weight=0.25,
    sparse_weight=0.75
):

    if not role_name or not role_name.strip():

        raise ValueError("role_name must be provided.")

    role_name = role_name.strip()

    # -----------------------------------------------------
    # STEP 1: COMPLETE MEDALLION PIPELINE
    # -----------------------------------------------------

    candidate_data = load_file(file_path)

    resume_data = candidate_data["resume_json"]

    fingerprint = candidate_data["fingerprint"]

    candidate_level = candidate_data["experience_data"]["candidate_level"]

    candidate_resume_json = json.dumps(
        resume_data,
        ensure_ascii=False
    )

    print("\nCANDIDATE INFORMATION")

    print("Fingerprint:", fingerprint)

    print("Retrieval Mode: Role + Experience Level")

    print("Target Role:", role_name)

    print("Candidate Level:", candidate_level)

    print(
        "Professional Experience:",
        candidate_data["experience_data"][
            "total_professional_experience_years"
        ],
        "years"
    )

    # -----------------------------------------------------
    # STEP 2: BUILD RETRIEVAL QUERIES
    # -----------------------------------------------------

    dense_query = build_dense_query(resume_data)

    sparse_query_data = build_sparse_query(resume_data)

    fts_query = build_fts_query(sparse_query_data)

    # -----------------------------------------------------
    # STEP 3: DENSE RETRIEVAL
    # -----------------------------------------------------

    retrieval_k = max(k * 3, 10)

    if dense_query:

        dense_results = dense_retrieve(

            query=dense_query,

            role_name=role_name,

            candidate_level=candidate_level,

            k=retrieval_k,

            fetch_k=20

        )

    else:

        dense_results = []

    # -----------------------------------------------------
    # STEP 4: SPARSE RETRIEVAL
    # -----------------------------------------------------

    if fts_query:

        sparse_results = sparse_retrieve(

            query=fts_query,

            role_name=role_name,

            candidate_level=candidate_level,

            k=retrieval_k

        )

    else:

        sparse_results = []


    

    # -----------------------------------------------------
    # DISPLAY DENSE AND SPARSE RANKINGS
    # -----------------------------------------------------

    display_retrieval_results(
        "DENSE RETRIEVAL - CHROMA",
        dense_results
    )

    display_retrieval_results(
        "SPARSE RETRIEVAL - BM25",
        sparse_results
    )


    # -----------------------------------------------------
    # STEP 5: RECIPROCAL RANK FUSION
    # -----------------------------------------------------

    rrf_results = reciprocal_rank_fusion(

        dense_results=dense_results,

        sparse_results=sparse_results,

        dense_weight=dense_weight,

        sparse_weight=sparse_weight

    )

    # -----------------------------------------------------
    # STEP 6: FETCH TOP K BENCHMARKS
    # -----------------------------------------------------

    final_results = fetch_benchmarks(
        rrf_results[:k]
    )

    # -----------------------------------------------------
    # STEP 7: DISPLAY RESULTS
    # -----------------------------------------------------

    print("\nRETRIEVAL RESULTS")

    print("Dense results:", len(dense_results))

    print("Sparse results:", len(sparse_results))

    print("Fused results:", len(rrf_results))

    print("Final benchmarks:", len(final_results))

    for rank, benchmark in enumerate(final_results, start=1):

        print(f"\nBenchmark Rank: {rank}")

        print("Resume ID:", benchmark["resume_id"])

        print("Candidate:", benchmark["candidate_name"])

        print("Candidate Level:", benchmark["candidate_level"])

        print("Role:", benchmark["role_name"])

        print("Human Score:", benchmark["human_score"])

        print("Quality:", benchmark["quality_label"])

        print("RRF Score:", round(benchmark["rrf_score"], 6))

        print("Evaluation Status:", benchmark["evaluation_status"])

    return final_results, candidate_resume_json


# =========================================================
# 9. TESTING
# =========================================================

if __name__ == "__main__":

    test_file = "dummy_resumes/weak_03.pdf"

    results, candidate_json = retrieve_role_level(

        file_path=test_file,

        role_name="AI Engineer",

        k=1

    )
