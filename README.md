# ResumeIQ 🎯
### AI-Powered Resume Intelligence & Candidate Evaluation

ResumeIQ is an AI-powered recruitment decision-support system that evaluates candidate resumes against specific job roles using **Retrieval-Augmented Generation (RAG), hybrid information retrieval, benchmark-based scoring, and Large Language Models (LLMs).**

Unlike conventional resume screeners that primarily rely on keyword matching, ResumeIQ combines semantic and lexical retrieval with human-rated benchmark resumes to generate structured, evidence-backed candidate assessments.

The system analyzes technical skills, projects, professional experience, role relevance, and education to identify candidate strengths, skill gaps, and areas for improvement.

Built with **Python, LangChain, Groq, FastAPI, Streamlit, ChromaDB, and SQLite**, ResumeIQ separates its frontend, API layer, document-processing pipeline, retrieval engine, and evaluation framework.

---

## ✨ Key Features

### 1. Intelligent Resume Processing
- Supports PDF, DOCX, TXT, DOC, RTF, JPG, JPEG, and PNG input formats.
- Extracts and structures relevant candidate information.
- Uses document fingerprinting to identify previously processed files.
- Applies extraction and processing limits to handle potentially resource-intensive documents.

### 2. Secure Document Ingestion
- File extension, MIME type, signature, and file-size validation.
- SHA-256 fingerprinting for document identification.
- ClamAV antivirus scanning.
- PDF and DOCX structural validation.
- OCR and extracted-text limits.
- Controlled processing time and memory resources.

### 3. Hybrid Information Retrieval
Combines two complementary retrieval approaches:

- **Dense Retrieval:** BGE embeddings with ChromaDB for semantic similarity.
- **Sparse Retrieval:** SQLite FTS5 with BM25 for keyword-based matching.
- **Reciprocal Rank Fusion (RRF):** Combines rankings from both retrieval systems.
- Retrieves relevant benchmark resumes based on the target role and candidate information.

### 4. Benchmark-Based Candidate Evaluation
- Uses human-rated benchmark resumes as scoring references.
- Compares candidates against relevant role-specific benchmarks.
- Generates structured assessments rather than relying exclusively on keyword overlap.
- Separates general resume evaluation from role-specific evaluation.

### 5. LLM-Powered Assessment
Uses LangChain and Groq to generate structured candidate evaluations, including:

- Overall candidate score.
- Technical skills assessment.
- Project and experience analysis.
- Role relevance.
- Candidate strengths.
- Identified skill gaps.
- Actionable improvement recommendations.

### 6. Multi-Layer Evaluation Framework
Dedicated evaluation suites assess different parts of the system:

- Retrieval evaluation.
- Generator evaluation.
- Pipeline evaluation.
- End-to-end application evaluation.

---

## 🏗️ System Architecture

ResumeIQ follows a layered architecture that separates the user interface, API boundary, application services, and core AI pipeline.

```text
                  USER
                    |
                    v
          STREAMLIT FRONTEND
               (app.py)
                    |
                    v
           FRONTEND API CLIENT
             (frontend/api.py)
                    |
             HTTP / REST API
                    |
                    v
             FASTAPI BACKEND
             (backend/main.py)
                    |
                    v
           REQUEST VALIDATION
           FILE HANDLING
                    |
                    v
          BACKEND SERVICE LAYER
           (backend/services.py)
                    |
                    v
           RESUME EVALUATION
               ENGINE
                    |
          +---------+----------+
          |                    |
          v                    v
    GENERAL EVALUATION   ROLE-SPECIFIC
                         EVALUATION
                              |
                              v
                       RESUME EXTRACTION
                              |
                              v
                       STRUCTURED DATA
                              |
                              v
                       HYBRID RETRIEVAL
                         /         \
                        v           v
                  DENSE SEARCH   BM25 SEARCH
                  (ChromaDB)   (SQLite FTS5)
                        \           /
                         v         v
                       RRF FUSION
                              |
                              v
                    RELEVANT BENCHMARKS
                              |
                              v
                      LLM ASSESSMENT
                     (LangChain/Groq)
                              |
                              v
                     STRUCTURED REPORT
                              |
                              v
                    FASTAPI JSON RESPONSE
                              |
                              v
                    STREAMLIT FRONTEND
```

### Architecture Components

| Layer | Responsibility |
|---|---|
| Streamlit Frontend | Resume uploads, evaluation selection, and result presentation |
| Frontend API Client | Sends multipart HTTP requests to the backend |
| FastAPI Backend | Exposes REST endpoints, validates requests, and manages uploaded files |
| Backend Services | Connects API requests to the existing evaluation engine |
| Document Processing | Validates, scans, extracts, and structures resume content |
| Retrieval Engine | Performs dense and BM25 retrieval with RRF |
| Assessment Engine | Generates structured candidate assessments using an LLM |
| Evaluation Framework | Measures retrieval, generation, pipeline, and application quality |

---

## 🔐 Document Security & Processing

ResumeIQ treats uploaded resumes as untrusted input and applies validation at multiple stages.

The document-processing pipeline includes:

| Security Layer | Purpose |
|---|---|
| Extension Validation | Rejects unsupported file formats |
| MIME Validation | Checks declared file content type |
| Signature Validation | Verifies file signatures |
| File Size Limits | Restricts oversized uploads |
| SHA-256 Fingerprinting | Generates document fingerprints |
| ClamAV Scanning | Scans files for known malware |
| Structural Validation | Checks PDF and DOCX document integrity |
| OCR Limits | Controls OCR processing and extracted content |
| Resource Controls | Restricts processing time and memory consumption |

The FastAPI layer performs preliminary request and upload validation, while the core extraction pipeline remains the authoritative document-processing and security boundary.

---

## 🔎 Hybrid Retrieval Architecture

ResumeIQ uses hybrid search to improve benchmark relevance by combining semantic understanding with exact keyword matching.

### Dense Retrieval
- Generates embeddings using BGE Sentence Transformers.
- Stores and searches vector representations using ChromaDB.
- Retrieves semantically similar benchmark resumes.

### Sparse Retrieval
- Uses SQLite FTS5 for full-text indexing.
- Applies BM25 ranking to identify keyword-relevant benchmarks.
- Emphasizes relevant technical skills, projects, and experience information.

### Reciprocal Rank Fusion

The rankings from both retrieval systems are combined using Reciprocal Rank Fusion:

\[
RRF(d)=\sum_{r \in R}\frac{1}{k+rank_r(d)}
\]

Where:
- \(d\) represents a retrieved document.
- \(R\) represents the retrieval systems.
- \(rank_r(d)\) is the document's rank in a retrieval system.
- \(k\) is the RRF constant.

This allows ResumeIQ to combine semantic and lexical relevance without requiring both retrieval systems to produce directly comparable scores.

---

## 📊 Evaluation & Benchmarking

ResumeIQ includes a dedicated evaluation framework designed to measure individual components as well as overall application quality.

### Retrieval Evaluation

| Metric | Purpose |
|---|---|
| Recall@K | Measures relevant benchmark retrieval |
| Precision@K | Measures relevance among retrieved results |
| MRR | Measures the ranking position of relevant results |
| NDCG@K | Measures ranking quality with relevance information |
| Latency | Measures retrieval execution time |

<text>**Current retrieval results**</text>

- Recall@5: **92.59%**
- MRR: **0.944**
- NDCG@5: **0.889**

### Generator Evaluation

Uses DeepEval and dedicated test datasets to assess generated candidate assessments.

| Metric | Result |
|---|---:|
| Faithfulness | 95.86% |
| Answer Relevancy | 100% |
| Mean Absolute Error against human scores | 1.62 |

### End-to-End Application Evaluation

The application evaluation suite assesses generated results against expected outputs.

| Metric | Result |
|---|---:|
| Correctness | 83.7% |
| Completeness | 100% |
| Tone | 91% |

*Results reflect the current evaluation datasets and test configurations. They are not guarantees of production performance or real-world recruitment accuracy.*

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| LLM | Groq |
| LLM Framework | LangChain |
| Embeddings | Hugging Face Sentence Transformers, BGE |
| Vector Database | ChromaDB |
| Keyword Retrieval | SQLite FTS5, BM25 |
| Database | SQLite |
| Backend Framework | FastAPI |
| Frontend Framework | Streamlit |
| API Communication | REST APIs, HTTP, Multipart Form Data |
| Evaluation | DeepEval |
| Document Security | ClamAV |
| Version Control | Git, GitHub |

---

## 🔌 API Architecture

ResumeIQ uses FastAPI as the backend API layer, allowing the Streamlit frontend to communicate with the evaluation engine through HTTP requests.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API root endpoint |
| GET | `/health` | Health-check endpoint |
| POST | `/evaluate` | Processes and evaluates an uploaded resume |

### Resume Evaluation API

**Endpoint:**

```http
POST /evaluate
```

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| file | File | Yes | Candidate resume |
| evaluation_type | String | Yes | `general` or `role_specific` |
| role_name | String | Conditional | Target role for role-specific evaluation |

**Example request using Python:**

```python
import requests

url = "http://127.0.0.1:8000/evaluate"

with open("resume.pdf", "rb") as file:
    response = requests.post(
        url,
        files={
            "file": (
                "resume.pdf",
                file,
                "application/pdf"
            )
        },
        data={
            "evaluation_type": "role_specific",
            "role_name": "AI Engineer"
        },
        timeout=300
    )

response.raise_for_status()

result = response.json()
print(result)
```

### API Documentation

FastAPI automatically provides interactive API documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 📂 Project Structure

```text
ResumeIQ/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── services.py
│   └── validation.py
│
├── frontend/
│   ├── __init__.py
│   ├── api.py
│   └── validation.py
│
├── .streamlit/
│   └── config.toml
│
├── evals/
│   ├── test_resumes/
│   ├── application_eval.py
│   ├── eval_suite.py
│   ├── generator_evaluator.py
│   ├── pipeline_evaluator.py
│   ├── retriever_eval.py
│   └── golden_datasets/
│
├── storage/
│   ├── chroma_db/
│   └── benchmarks.db
│
├── extraction.py
├── retriever.py
├── generator.py
├── benchmarks.py
├── resume_utils.py
├── structured_extraction.py
├── app.py
│
├── requirements.txt
├── .env.example
└── .gitignore
```

*The structure above highlights the main application components; additional evaluation scripts, datasets, and supporting modules may exist.*

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or later (use the version supported by your dependencies).
- Git.
- Groq API key.
- ClamAV installed and configured for malware scanning.

### 1. Clone the repository

```bash
git clone https://github.com/OfficialHarshVerma/ResumeIQ.git

cd ResumeIQ
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

Keep API keys private. Never commit your `.env` file to GitHub.

### 5. Start the FastAPI backend

Open the first terminal:

```bash
uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Verify the backend using:

```text
http://127.0.0.1:8000/health
```

### 6. Start the Streamlit frontend

Open a second terminal with the same virtual environment activated:

```bash
streamlit run app.py
```

The application will be available at the local Streamlit URL displayed in the terminal.

**Important:** The FastAPI backend must be running before submitting a resume through the Streamlit frontend.

---

## 🚧 Current Development Status

ResumeIQ currently includes:

- Streamlit-based frontend.
- FastAPI backend with REST API integration.
- General and role-specific resume evaluation.
- Multi-layer document validation and security controls.
- Hybrid dense and BM25 retrieval with RRF.
- Benchmark-based LLM candidate assessment.
- Dedicated retrieval, generator, and application evaluation suites.

The project is actively evolving, with ongoing improvements to API reliability, validation, application security, and overall system robustness.

---

## 🗺️ Future Roadmap

- Job-description-driven candidate matching.
- Recruiter dashboard for candidate comparison and shortlisting.
- Improved benchmark calibration and scoring consistency.
- Expanded adversarial document and API security testing.
- API authentication, authorization, and rate limiting.
- Containerization and deployment.
- Improved monitoring, logging, and application observability.
- More extensive performance and concurrent-request testing.

---

## 👨‍💻 Author

**Harsh Verma**

B.Tech – Electronics and Communication Engineering

Interested in Generative AI, LLM Engineering, Retrieval-Augmented Generation, and AI Agent Development.

GitHub: [OfficialHarshVerma](https://github.com/OfficialHarshVerma)
