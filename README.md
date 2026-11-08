# ResumeIQ 🎯
### AI-Powered Resume Intelligence & Candidate Evaluation

TalentLens is an AI-powered resume evaluation and job-matching system designed to analyze candidate profiles against specific job roles. It combines **Retrieval-Augmented Generation (RAG), hybrid search, benchmark-based scoring, and LLM-driven evaluation** to produce structured, evidence-based candidate assessments.

The goal is to go beyond simple keyword matching and provide meaningful insights into a candidate's technical skills, practical projects, professional experience, and suitability for a target role.

## ✨ Key Features

- **Resume Parsing:** Extracts structured information from candidate resumes.
- **Hybrid Retrieval:** Combines semantic search and BM25 keyword retrieval to identify relevant benchmark resumes.
- **Reciprocal Rank Fusion (RRF):** Combines results from multiple retrieval methods.
- **Benchmark-Based Evaluation:** Uses human-rated benchmark resumes to provide scoring calibration references.
- **LLM-Based Assessment:** Evaluates candidates across multiple criteria and generates structured assessments.
- **Evidence-Based Reasoning:** Identifies candidate strengths, skill gaps, and actionable improvements.
- **Evaluation Suite:** Independently evaluates retrieval, generation, pipeline performance, and overall application quality.

## 🏗️ System Architecture

```text
             Candidate Resume
                    |
                    v
              Resume Parsing
                    |
                    v
          Structured Resume Data
                    |
                    v
             Hybrid Retrieval
              /           \
             v             v
       Dense Search     BM25 Search
             \             /
              v           v
          Reciprocal Rank Fusion
                    |
                    v
          Relevant Benchmark Resumes
                    |
                    v
             LLM-Based Scoring
                    |
                    v
          Structured Candidate Report
                    |
          +---------+---------+
          |         |         |
       Final     Strengths  Skill Gaps
       Score               & Improvements
```

## 🧠 Evaluation Methodology

TalentLens includes a dedicated evaluation suite to measure the reliability of its individual components.

| Component | Evaluation |
|---|---|
| Retriever | Recall@K, Precision@K, MRR, NDCG@K, latency |
| Generator | Faithfulness, answer relevancy and scoring quality |
| Pipeline | End-to-end performance and evaluation metrics |
| Application | Correctness, completeness and tone |

The system also uses human-rated benchmark resumes to assess the consistency of LLM-generated candidate scores.

## 🛠️ Tech Stack

- **Language:** Python
- **LLM:** Groq
- **LLM Framework:** LangChain
- **Embeddings:** Hugging Face Sentence Transformers
- **Vector Database:** ChromaDB
- **Keyword Retrieval:** SQLite FTS5 (BM25)
- **Database:** SQLite
- **Evaluation:** DeepEval
- **Interface:** Streamlit

## 📂 Project Structure

```text
TalentLens/
│
├── evals/
│   ├── test_resumes/
│   │
│   ├── application_eval.py
│   ├── eval_suite.py
│   │
│   ├── generator_evaluator.py
│   ├── generator_export.py
│   ├── gen_eval_temp.py
│   ├── generator_golden_dataset.csv
│   ├── generator_golden_dataset.json
│   ├── generated_outputs.json
│   │
│   ├── pipeline_evaluator.py
│   ├── pipeline_export.py
│   ├── pipeline_outputs.json
│   │
│   ├── retriever_eval.py
│   ├── retriever_golden_dataset.json
│   ├── resume_retrieval_golden_dataset_v0.2.json
│   │
│   └── evaluation_results.csv
|
├── storage/
│   ├── chroma_db/
│   └── benchmarks.db
│
├── user.py
├── retriever_eval.py
├── benchmarks.py
├── resume_utilities.py
├── test.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*Project structure may evolve as development progresses.*

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/talentlens.git
cd talentlens
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the application

```bash
streamlit run app.py
```

## 🚧 Development Status

TalentLens is actively under development. The current implementation focuses on building and evaluating a reliable resume retrieval and scoring pipeline. The architecture and feature set may evolve as the project is redesigned and expanded.

## 🎯 Future Roadmap

- Job-description-based candidate matching.
- Improved benchmark-based scoring calibration.
- More robust resume parsing and structured extraction.
- Enhanced evaluation and error analysis.
- Recruiter-focused candidate comparison and shortlisting.
- API integration and persistent application deployment.

## 👨‍💻 Author

**Harsh Verma**

B.Tech – Electronics and Communication Engineering

Practical in Artificial Intelligence, Machine Learning, and Generative AI.
