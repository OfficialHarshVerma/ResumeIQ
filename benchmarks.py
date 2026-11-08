import sqlite3
from resume_utilities import load_pdf, resume_extractor
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from resume_utilities import load_pdf,build_resume_fields
from langchain_core.documents import Document




# CREATING STORAGES------------------------<<<<<<<<>>>>>>>
# behcmark_resumes and role_guides
connection = sqlite3.connect("storage/benchmarks.db")

connection.executescript("""
    CREATE TABLE IF NOT EXISTS benchmark_resumes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint   TEXT NOT NULL UNIQUE,
    role_name     TEXT NOT NULL,
    resume_json   TEXT NOT NULL,
    human_score   REAL NOT NULL CHECK (human_score >= 0 AND human_score <= 10),
    score_reason  TEXT NOT NULL,
    quality_label TEXT NOT NULL CHECK (quality_label IN ('strong', 'average', 'weak'))
);

    CREATE TABLE IF NOT EXISTS role_guides (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name  TEXT NOT NULL UNIQUE,
    summary    TEXT NOT NULL
);

    
    CREATE VIRTUAL TABLE IF NOT EXISTS benchmark_fts USING fts5(
        fingerprint UNINDEXED,
        skills,
        experience,
        projects,
        certifications
    );


"""
)
connection.commit()
connection.close()



#embedding model
# vector store

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="storage/chroma_db",
    collection_name="resume_skilks"
)







# benchmark_query -------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ADDs resume entry as a row in the benchmark_resumes
def benchmark_query(fingerprint,role_name, resume_json, human_score, score_reason, quality_label):
    connection = sqlite3.connect("storage/benchmarks.db")

    try:

        connection.execute("""
        
            INSERT OR IGNORE INTO benchmark_resumes
            ( fingerprint,role_name, resume_json, human_score, score_reason, quality_label )
            VALUES (?,?,?,?,?,?)
        """,(fingerprint,role_name,resume_json,human_score,score_reason,quality_label))

        
        connection.commit()

    finally:
        connection.close()

# check if a row with the fingerprint already exists or not
def check_fingerprint(fingerprint):

    connection = sqlite3.connect("storage/benchmarks.db")

    result = connection.execute("""
        SELECT *
        FROM benchmark_resumes
        WHERE fingerprint = ?
    """,(fingerprint,)
    ).fetchone()

    connection.close()

    if result:
        print("Fingerprint Exists")
        return True
    else:
        print("Fingerprint not there adding it now...")
        return False 

# Fetches the resume_json from the benchmarks_resumes table of  a particular fingerprint
def fetch_resume_json(fingerprint):
    connection = sqlite3.connect("storage/benchmarks.db")

    resume = connection.execute("""
        SELECT resume_json
        FROM benchmark_resumes
        WHERE fingerprint = ?
    """,(fingerprint,)).fetchone()

    connection.close()
    print("Resume Json Found!")
    return resume[0] if resume else None

temp = "Pandas, NumPy, Scikit-Learn, Exploratory Data Analysis, Supervised/Unsupervised Learning, Regression, Classification, Python, SQL, LangChain, LangGraph, RAG, Hugging Face, Jupyter Notebook, Git, GitHub, MySQL, DSA, OOP, DBMS, Whisper, PaddleOCR, FAISS, PySide6, Streamlit, BGE embeddings"




# creating embedding and storing to the ChromaDB ---------------->>>>>>>>>>>>>>>>>>>

def add_embeddings(fingerprint, resume_data):

    resume_json = json.loads(resume_data)

    # 1. Skills
    technical_skills = resume_json['skills']['technical_skills']
    soft_skills = resume_json['skills']['soft_skills']

    skills_text = (
        f"Technical skills: {', '.join(technical_skills)}. "
        f"Soft skills: {', '.join(soft_skills)}."
    )

    # 2. Experience
    experience_text = []

    for exp in resume_json['experience']:
        experience_text.append(
            f"Job title: {exp.get('job_title', '')}. "
            f"Company: {exp.get('company', '')}. "
            f"Employment type: {exp.get('employment_type', '')}. "
            f"Location: {exp.get('location') or ''}. "
            f"Duration: {exp.get('start_date', '')} to {exp.get('end_date', '')}. "
            f"Responsibilities: {', '.join(exp.get('responsibilities', []))}."
        )

    experience_text = " ".join(experience_text)

    # 3. Projects
    projects_text = []

    for project in resume_json['projects']:
        projects_text.append(
            f"Project: {project.get('project_name', '')}. "
            f"Description: {project.get('description', '')}. "
            f"Technologies: {', '.join(project.get('technologies', []))}. "
            f"Key features: {', '.join(project.get('key_features', []))}."
        )

    projects_text = " ".join(projects_text)

    # 4. Certifications
    certifications = resume_json['skills']['certifications']
    certifications_text = ', '.join(certifications)

    # Combine all sections
    resume_text = f"""
            Skills:
            {skills_text}

            Experience:
            {experience_text}

            Projects:
            {projects_text}

            Certifications:
            {certifications_text}
        """

    print(fingerprint, resume_text)



    # For adding new embeddings
    # vector_store.add_texts(
    #     texts=[resume_text],
    #     ids=[fingerprint],
    #     metadatas=[{"fingerprint": fingerprint}]
    # )

    # print("Successfully stored your data in ChromaDB")



    # For updating the embeddings
    doc = Document(
        page_content=resume_text,
        metadata={"fingerprint": fingerprint}
    )

    vector_store.update_documents(
        ids=[fingerprint],
        documents=[doc]
    )

    print("Successfully updated embedding in ChromaDB")



def add_sparse_embedding(fingerprint, resume_json):

    if check_sparse_fingerprint(fingerprint):
        print("Fingerprint already exists in FTS5.")
        return

    fields = build_resume_fields(json.loads(resume_json))

    connection = sqlite3.connect("storage/benchmarks.db")

    connection.execute("""
        INSERT INTO benchmark_fts (
            fingerprint,
            skills,
            experience,
            projects,
            certifications
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        fingerprint,
        fields["skills"],
        fields["experience"],
        fields["projects"],
        fields["certifications"]
    ))

    connection.commit()
    connection.close()



def check_sparse_fingerprint(fingerprint):

    connection = sqlite3.connect("storage/benchmarks.db")

    result = connection.execute("""
        SELECT *
        FROM benchmark_fts
        WHERE fingerprint = ?
        LIMIT 1
    """, (fingerprint,)).fetchone()

    connection.close()

    if result:
        print("Sparse Embedding already exists!!!")
        return True
    else:
        print("Sparse Doesn't exists!!")
        return False



# checks if embedding already exists or not int he Chrome DB
def embedding_exists(fingerprint):

    response = vector_store.get(
        ids = [fingerprint]
    )
    print("embedding exits!!!")
    return len(response["ids"]) > 0




# Add benchamrks --------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# the whole process or adding resume entry to both benchmark_resumes and chromaDB
def add_benchmarks(Path,role_name,human_score,score_reason,quality_label):
    path = Path


    # got the resume text and fingerprint
    # adding embeddings with fingerprints in the chroma db
    resume_text,fingerprint = load_pdf(path)


    # checking if the person already exists or not
    if not check_fingerprint(fingerprint=fingerprint):

        # resume json
        #
        resume_json = resume_extractor(resume_text=resume_text)
        benchmark_query(
        fingerprint=fingerprint,
        role_name=role_name,
        resume_json=resume_json,
        human_score=human_score,
        score_reason=score_reason,
        quality_label=quality_label
        )
    else:
        resume_json = fetch_resume_json(fingerprint)
        print("Resume Json Fetched Successfully!!!")


    # For updating 
    # add_embeddings(fingerprint=fingerprint,resume_data=resume_json)
    

    # For new ones
    if embedding_exists(fingerprint=fingerprint):
        print("embedding already exists")
    else:
        print("calling embeddings..")
        add_embeddings(fingerprint=fingerprint,resume_data=resume_json)


    if check_sparse_fingerprint(fingerprint=fingerprint):
        print("Sparse Found!!!")
    else:
        print("creating the sparks...")
        add_sparse_embedding(fingerprint=fingerprint,resume_json=resume_json)


    print("check to see if it works or not")


# Add Benchmark resume------------------>>>>>>>>>>>>>>>>>>>>>>
# calling the resume
# add_benchmarks(
#     Path="test_resume.pdf",
#     role_name="AI Engineer",
#     human_score=8.0,
#     score_reason="very lovely guy",
#     quality_label="strong"
#     )



from pathlib import Path
import pandas as pd
# to load the multiple pdfs together--------->>>>>>>>>>>>>>>>>
def add_multi_benchmarks(folder_path):
    
    metadata = pd.read_csv(folder_path +"/benchmark_metadata.csv")

    for pdf_path in Path(folder_path).glob("*.pdf"):

        # Find metadata for the current PDF
        row = metadata.loc[metadata["file"] == pdf_path.name]

        if row.empty:
            print(f"No metadata found for {pdf_path.name}")
            continue

        row = row.iloc[0]

        add_benchmarks(
            Path=pdf_path,
            role_name=row["role_name"],
            human_score=float(row["human_score"]),
            score_reason=row["score_reason"],
            quality_label=row["quality_label"]
        )

        print(f"Added: {pdf_path.name}")




# add_multi_benchmarks("dummy_resumes")




# ROLE GUIDE ----------------------<<<<<<<<<<>>>>>>>
# returns Role Guide if exists
def  get_role_guide(role_name):

    connection = sqlite3.connect("storage/benchmarks.db")

    role_summary = connection.execute("""
        SELECT summary
        FROM role_guides
        WHERE role_name = ?
    """,(role_name,)).fetchone()

    connection.close()

    return role_summary[0] if role_summary else None


def add_role_guide(role_name,role_summary):

    # check role_summary
    if not get_role_guide(role_name=role_name):
        
        connection = sqlite3.connect("storage/benchmarks.db")

        connection.execute("""
            INSERT INTO role_guides
            ( role_name, summary)
            VALUES (?,?)
        """,(role_name,role_summary))

        connection.commit()

        connection.close()

        if get_role_guide(role_name=role_name):
            print("Role Guide Created Successfully")
        else:
            print("Something went wrong!")

    else:
        print("role guide already exists")
    

role_name = "AI Engineer"

role_summary = """
An AI Engineer designs, develops, and deploys artificial intelligence
systems. Responsibilities include building machine learning models,
working with deep learning frameworks, developing LLM-powered applications,
implementing RAG pipelines, integrating APIs, and evaluating model
performance. Essential skills include Python, SQL, machine learning,
deep learning, NLP, LangChain, vector databases, and model deployment.
"""
# calling the role_guide adder
# add_role_guide(role_name=role_name,role_summary=role_summary)



import sqlite3

def update_role_guide(role_name, new_role_guide):
    conn = sqlite3.connect("storage/benchmarks.db")
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE role_guides
        SET summary = ?
        WHERE role_name = ?
    """, (new_role_guide, role_name))

    updated = cursor.rowcount

    conn.commit()
    conn.close()

    if updated:
        print(f"Role guide updated for: {role_name}")
    else:
        print(f"No role guide found for: {role_name}")

    return updated > 0

new_role_guide = role_guide = """
ROLE: AI Engineer

PURPOSE OF THE ROLE:
Designs, builds, evaluates, and deploys AI systems, especially machine learning models and LLM-powered applications such as RAG pipelines, chatbots, and NLP tools.

CORE REQUIREMENTS (these drive the scores; solid evidence(especially experience) in most of them means the candidate meets the role's expectations):
1. Programming: strong Python for ML or application development.
2. Machine learning and deep learning fundamentals: training, evaluating, and improving models (for example scikit-learn, PyTorch, or TensorFlow).
3. Applied AI work: has built real AI applications, such as an NLP system, an LLM application, a RAG pipeline, a recommender, or a computer vision model.
4. Model or system evaluation: measures performance with suitable metrics and iterates based on results.
5. Deployment or integration: exposes a model or pipeline through an API, an app, or a cloud service.

IMPORTANT BUT NOT MANDATORY (strong evidence raises scores):
- SQL and data handling (pandas, data cleaning, feature engineering)
- LLM tooling (LangChain, LlamaIndex, prompt engineering, fine-tuning)
- Vector databases and embeddings (FAISS, Chroma, Pinecone, Weaviate)
- Retrieval techniques such as hybrid search and reranking
- Git and collaborative development

OPTIONAL BONUSES (never reduce a score if missing):
- Docker, Kubernetes, CI/CD, MLOps tools, cloud certifications
- Specific frameworks or vendors beyond the examples above
- Research papers, open-source contributions, competition results
- Postgraduate degrees

EQUIVALENT TOOLS AND TRANSFERABLE EXPERIENCE:
Treat comparable tools as equivalent. Examples: LangChain, LlamaIndex, and a custom pipeline; FAISS, Chroma, Pinecone, and pgvector; PyTorch and TensorFlow; AWS, GCP, and Azure. Related backgrounds (data science, data engineering, software engineering with ML projects, NLP research) earn real but somewhat reduced credit for the criteria they overlap with.

HOW TO JUDGE DEPTH ACROSS EXPERIENCE LEVELS:
Judge each candidate by the depth, scope, and impact of the work shown, not by years of experience or job titles alone. Projects, internships, freelance work, and full-time roles all count as evidence. Stronger evidence looks like: clear ownership, systems built end to end, measurable results, scale or production use, and design decisions the candidate can explain. Candidates with more experience are expected to show more of these signals in order to reach the top scores, but a candidate with fewer years who shows them should score equally high.

HOW EACH SCORING CRITERION MAPS TO THIS ROLE:
- Technical skills: breadth and depth across the core requirements, shown through use and not just listing.
- Projects: AI projects with a clear goal, a described implementation, and results or evaluation. A few well-described projects are enough for a high score.
- Relevant experience: internships, jobs, or freelance work involving ML, data, or AI engineering. Judge the relevance and impact of the work, not the length of the history.
- Role-specific knowledge: understanding of how ML and LLM systems are built, evaluated, and deployed (RAG, embeddings, metrics, model lifecycle), shown through how the candidate describes their work.
- Education/certifications: a relevant degree or relevant courses and certifications. Self-taught candidates with strong evidence elsewhere should not be penalised heavily here.

WHAT A STRONG CANDIDATE LOOKS LIKE:
Has built and evaluated AI systems end to end, can explain what they implemented and what results they got, and shows working knowledge of Python, ML or deep learning, and at least one applied area such as NLP or LLM applications.

WHAT A CANDIDATE MEETING EXPECTATIONS LOOKS LIKE:
Solid Python, sound ML fundamentals, and relevant projects or work experience with clear implementation. Such a candidate should score around 7 on the main criteria even if SQL, deployment, or vector database experience is limited.

WHAT SHOULD NOT LOWER A SCORE:
Missing optional tools, not having used a specific framework, lack of cloud certifications, or a shorter work history when the evidence of skill is strong.
"""


# update_role_guide(new_role_guide=new_role_guide,role_name="AI Engineer")





# python benchmarks.py