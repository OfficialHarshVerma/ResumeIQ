import pymupdf
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from pydantic import BaseModel,Field
from typing import Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import json
import hashlib
from benchmarks import vector_store,get_role_guide,fetch_resume_json
from resume_utilities import load_pdf,resume_extractor,build_resume_text
import sqlite3
import re
import time






load_dotenv()


model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_retries=2,
        timeout= 30,
        max_tokens=3000
        )


# retriever ------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>------------------------
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}
)

def build_bm25_query(resume_data):

    query = {
        "skills": [],
        "projects": [],
        "experience": [],
        "certifications": []
    }
    # 1. Technical skills only
    skills = resume_data.get("skills", {})

    technical_skills = skills.get("technical_skills", [])

    query["skills"].extend(technical_skills)

    # 2. Projects
    for project in resume_data.get("projects", []):

        query["projects"].append(project.get("project_name", ""))

        query["projects"].extend(
            project.get("technologies", [])
        )

        query["projects"].append(project.get("description", ""))

        query["projects"].extend(
            project.get("key_features", [])
        )

    # 3. Experience
    for exp in resume_data.get("experience", []):

        query["experience"].append(
            exp.get("job_title", "")
        )

        query["experience"].extend(
            exp.get("responsibilities", [])
        )

    # 4. Certifications
    query["certifications"].extend(
        skills.get("certifications", [])
    )

    # Remove empty values and duplicates
    for category in query:

        query[category] = list(dict.fromkeys(
            term.strip()
            for term in query[category]
            if isinstance(term, str) and term.strip()
        ))

    return query

def bm25_search(query, k=7):

    query_parts = []

    column_map = {
        "skills": "skills",
        "projects": "projects",
        "experience": "experience",
        "certifications": "certifications"
    }

    for category, column in column_map.items():

        terms = query.get(category, [])

        if not terms:
            continue

        # Extract unique words from this category
        words = list(dict.fromkeys(
            word
            for term in terms
            for word in re.findall(r'\w+', term.lower())
        ))

        if words:
            category_query = " OR ".join(
                f'"{word}"' for word in words
            )

            query_parts.append(
                f'{column} : ({category_query})'
            )

    if not query_parts:
        return []

    fts_query = " OR ".join(query_parts)

    connection = sqlite3.connect("storage/benchmarks.db")

    results = connection.execute("""
        SELECT fingerprint,
               bm25(
                   benchmark_fts,
                   0.0,
                   3.0,
                   3.5,
                   3.5,
                   0.5
               ) AS score
        FROM benchmark_fts
        WHERE benchmark_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """, (fts_query, k)).fetchall()

    connection.close()

    return results


def fetch_sparse_resumes(sparse_results):

    retrieved_resumes = []

    connection = sqlite3.connect("storage/benchmarks.db")

    try:
        for fingerprint, score in sparse_results:

            result = connection.execute("""
                SELECT
                    fingerprint,
                    resume_json,
                    human_score,
                    score_reason,
                    quality_label
                FROM benchmark_resumes
                WHERE fingerprint = ?
            """, (fingerprint,)).fetchone()

            if result:
                retrieved_resumes.append({
                    "fingerprint": result[0],
                    "rrf_score": score,
                    "resume": json.loads(result[1]),
                    "human_score": result[2],
                    "score_reason": result[3],
                    "quality_label": result[4]
                })

    finally:
        connection.close()

    return retrieved_resumes


def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    k=60,
    dense_weight=0.25,
    sparse_weight=0.75
):
    scores = {}

    # Process dense and sparse rankings
    for results, weight in [
        (dense_results, dense_weight),
        (sparse_results, sparse_weight)
    ]:

        for rank, (fingerprint, _) in enumerate(results, start=1):

            if fingerprint not in scores:
                scores[fingerprint] = 0

            scores[fingerprint] += weight / (k + rank)

    # Sort by RRF score in descending order
    ranked_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results



def retrieve(file_path, k=7):

    # 1. Load candidate resume and get structured JSON
    resume_data, fingerprint = load_pdf(file_path=file_path)

    candidate_resume_json = json.dumps(resume_data)

    # 2. Generate combined resume text for dense retrieval
    combined_text = build_resume_text(resume_data)

    # 3. Dense retrieval - ChromaDB
    dense_docs = retriever.invoke(combined_text)[:k]

    # Convert dense documents into (fingerprint, score) tuples
    dense_results = [
        (doc.metadata["fingerprint"], None)
        for doc in dense_docs
    ]

    # 4. Sparse retrieval - SQLite FTS5
    bm25_query = build_bm25_query(resume_data)
    sparse_results = bm25_search(query=bm25_query, k=k)

    # print("\nDENSE RANKING:")
    # for rank, doc in enumerate(dense_docs, start=1):
    #     print(rank, doc.metadata["fingerprint"])

    # print("\nBM25 RANKING:")
    # for rank, (fingerprint, score) in enumerate(sparse_results, start=1):
    #     print(rank, fingerprint, score)

    # 5. Reciprocal Rank Fusion
    rrf_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results
    )

    # 6. Fetch complete resumes using fused ranking
    final_results = fetch_sparse_resumes(rrf_results[:1])

    for benchmark in final_results:
        print("Name:", benchmark["resume"]["personal_info"]["name"])
        print("Human Score:", benchmark["human_score"])
        print("Quality:", benchmark["quality_label"])
        print("Reason:", benchmark["score_reason"])

    return final_results, candidate_resume_json


# retrieve(file_path="evals/test_resumes/average_08.pdf")



# For running the function for all teh resumes in the test_resume folder
from pathlib import Path

def run_all_retrievals():
    test_folder = Path("evals/test_resumes")

    resume_files = sorted(test_folder.glob("*.pdf"))

    for resume_path in resume_files:
        print("\n" + "=" * 60)
        print(f"Evaluating: {resume_path.name}")
        print("=" * 60)

        retrieve(file_path=str(resume_path))

# run_all_retrievals()


# For Generator function ---------------------<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>------------------
system_prompt = system_prompt = """## 1. Evaluation Criteria and Weights

Score each criterion independently from 0–10.

- Technical skills: 30%
- Projects/practical implementation: 30%
- Relevant experience: 30%
- Role-specific knowledge: 5%
- Education/certifications: 5%

## 2. General Scoring Scale

- 0–2: Little or no relevant evidence.
- 3–4: Weak foundational knowledge or superficial practical exposure.
- 5–6: Average candidate with relevant foundational skills and genuine practical implementation, but limited depth.
- 7: Competent candidate who meets the core expectations, not an advanced or production-level expert.
- 8: Strong candidate demonstrating substantial implementation depth and independent work.
- 9–10: Excellent candidate demonstrating exceptional technical depth, ownership, problem-solving, and results.

## 3. Criterion Evaluation Steps

For EACH criterion, follow these steps independently:

1. Identify all relevant resume evidence.
2. Assess its relevance, practical application, and implementation depth.
3. Evaluate the evidence against the corresponding criterion rubric.
4. Identify meaningful strengths and limitations.
5. Assign a score using the general scoring scale.
6. Verify that the score is justified by the evidence.

Do not let one criterion's strengths or weaknesses determine another criterion's score.

## 4. Criterion-Specific Rubrics

**Technical skills (30%):** Evaluate practical proficiency in relevant programming languages, frameworks, libraries, and tools. Distinguish listed skills from demonstrated technical ability.

**Projects (30%):** Evaluate project relevance, implementation, complexity, problem-solving, and evaluation. Give credit to genuine basic implementations without requiring production deployment or advanced metrics.

**Experience (30%):** Evaluate relevant responsibilities, practical technical work, ownership, and contributions. Distinguish professional employment from academic and personal projects, while recognizing transferable experience.

**Role knowledge (5%):** Evaluate understanding of core concepts, workflows, and technical decisions demonstrated through actual implementation.

**Education/certifications (5%):** Evaluate relevant academic foundations and structured learning. Do not automatically penalize unrelated degrees or missing optional certifications.

## 5. Benchmark-Based Calibration

The benchmark resumes are historical examples evaluated by human reviewers. Each benchmark contains:
- `resume`: The candidate's resume information.
- `human_score`: The human-assigned overall score (0–10).
- `score_reason`: The explanation behind the human evaluation.
- `quality_label`: The human-assigned category (strong, average, or weak).
- `rrf_score`: Retrieval relevance score. This is NOT a candidate quality score.

Use these labelled examples as reference points to calibrate your evaluation.

Follow these rules:

1. **Study the evaluation examples:** Examine the resume evidence alongside its human score, quality label, and score reason. Understand what level of implementation, experience, and technical depth corresponded to that evaluation.

2. **Compare implementation depth:** Identify similarities and differences between the candidate and benchmark resumes in terms of actual skills, projects, responsibilities, ownership, and demonstrated outcomes.

3. **Use human scores as calibration references:** A benchmark with a similar level of demonstrated ability can help establish an appropriate scoring range. Do not blindly copy its score or calculate the candidate's score by averaging benchmark scores.

4. **Understand the reasoning behind scores:** Pay attention to why a benchmark received its score. For example, distinguish basic API integration from actual model development, or genuine applied ML implementation from superficial exposure.

5. **Evaluate independently:** The candidate must be assessed against the role guide and evaluation criteria, not ranked merely by similarity to a benchmark.

6. **Do not overgeneralize:** A similar resume structure, technology list, or project title does not establish equivalent ability. Compare the actual implementation evidence.

7. **Handle conflicting examples carefully:** If similar benchmarks have different human scores, examine their score reasons and evidence to understand the difference. Do not assume that every benchmark label or score is perfectly consistent.

8. **Do not invent benchmark evidence:** Only use the information actually present in the retrieved benchmark data.

The purpose of benchmarks is to improve scoring consistency and human-rating alignment, not to restrict candidates to the backgrounds of previously evaluated resumes.

## 6. Final Score

Calculate using the fixed weights:

final_score = (technical_skills × 0.20) + (projects × 0.20) + (experience × 0.50) + (role_knowledge × 0.05) + (education_certifications × 0.05)

Round to one decimal place.

## 7. Output

- reasoning: 3–5 sentences.
- strengths: 2–5 evidence-based items.
- skill_gaps: Only meaningful core deficiencies.
- improvements: 2–4 actionable suggestions relevant to the candidate's specialization.

Return valid JSON only, using this schema:
{{
  "role": "string",
  "final_score": 0.0,
  "criteria_scores": {{
    "technical_skills": 0.0,
    "projects": 0.0,
    "experience": 0.0,
    "role_knowledge": 0.0,
    "education_certifications": 0.0
  }},
  "reasoning": "string",
  "strengths": ["string"],
  "skill_gaps": ["string"],
  "improvements": ["string"]
}}
"""

# main part to get the score out of the llm
scoring_prompt = ChatPromptTemplate([
    ( "system",system_prompt),
    ("human","""
        Role: {role_name}

        Role Guide:
        {role_guide}

        Candidate Resume JSON:
        {candidate_resume_json}

        Benchmark Resume JSON:
        {benchmark_resume_json}
        """)
])

# output structure for this


class CriteriaScores(BaseModel):
    technical_skills: float = Field(ge=0, le=10)
    projects: float = Field(ge=0, le=10)
    experience: float = Field(ge=0, le=10)
    role_knowledge: float = Field(ge=0, le=10)
    education_certifications: float = Field(ge=0, le=10)


class ResumeScore(BaseModel):
    role: str
    final_score: float = Field(ge=0, le=10)
    criteria_scores: CriteriaScores
    reasoning: str
    strengths: list[str]
    skill_gaps: list[str]
    improvements: list[str]




def calculate_final_score(criteria_scores):
    weights = {
        "technical_skills": 0.30,
        "projects": 0.30,
        "experience": 0.30,
        "role_knowledge": 0.05,
        "education_certifications": 0.05
    }

    final_score = sum(
        getattr(criteria_scores, criterion) * weight
        for criterion, weight in weights.items()
    )

    return round(final_score, 2)



# scoring structured llm
scoring_llm = model.with_structured_output(ResumeScore,method="json_mode")


scoring_chain = scoring_prompt | scoring_llm 

# get Resume Score ---------------------------------------<<<<<<<>>>>>>>

# Get Resume Score ---------------------------------------


def generate(file_path, role_name, return_context=False):

    # 1. Retrieve top 3 benchmark resumes
    docs, candidate_resume_json = retrieve(
        file_path=file_path,
        k=7
    )

    if not docs:
        raise ValueError("No benchmark resumes were retrieved.")

    # 2. Prepare all three benchmark resumes
    benchmark_resumes = []

    for rank, doc in enumerate(docs, start=1):

        benchmark_resumes.append({
            "rank": rank,
            "rrf_score": doc["rrf_score"],
            "fingerprint": doc["fingerprint"],
            "resume": doc["resume"],
            "human_score": doc["human_score"],
            "score_reason": doc["score_reason"],
            "quality_label": doc["quality_label"]
        })

        print(f"\nBenchmark Rank: {rank}")
        print("Name:", doc["resume"]["personal_info"]["name"])
        # print("RRF Score:", doc["rrf_score"])

    # 3. Get role guide
    role_guide = get_role_guide(role_name=role_name)

    # 4. Serialize all benchmark resumes
    benchmark_resume_json = json.dumps(
        benchmark_resumes,
        indent=2
    )

    # prompt_value = scoring_prompt.format_prompt(
    #     role_name=role_name,
    #     role_guide=role_guide,
    #     candidate_resume_json=candidate_resume_json,
    #     benchmark_resume_json=benchmark_resume_json
    # )

    # print(prompt_value.to_string())

    # 5. Generate score
    result = scoring_chain.invoke({
        "role_name": role_name,
        "role_guide": role_guide,
        "candidate_resume_json": candidate_resume_json,
        "benchmark_resume_json": benchmark_resume_json
    })


    result.final_score = calculate_final_score(result.criteria_scores)

    print("\n========== FINAL RESULT ==========")
    print(result)

    if return_context:

        retrieval_context = [
            "Candidate resume:\n" + candidate_resume_json,
            "Role guide:\n" + str(role_guide),
            "Retrieved Benchmark Resumes:\n" + benchmark_resume_json
        ]

        return result, retrieval_context

    return result


pdf_files = [
    # Path("evals/test_resumes/strong_08.pdf"),
    # Path("evals/test_resumes/strong_09.pdf"),
    Path("evals/test_resumes/strong_10.pdf"),
    # Path("evals/test_resumes/weak_08.pdf"),
    Path("evals/test_resumes/weak_09.pdf"),
    # Path("evals/test_resumes/weak_10.pdf"),
    Path("evals/test_resumes/average_08.pdf"),
    # Path("evals/test_resumes/average_09.pdf"),
    Path("evals/test_resumes/average_10.pdf"),
]

for path in pdf_files:
    generate(file_path=path,role_name="AI Engineer")

