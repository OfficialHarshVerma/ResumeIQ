
import json
from typing import Dict
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

from extraction import load_file
from benchmarks import get_role_guide
from retriever import retrieve_role_level


load_dotenv()


# =========================================================
# 1. LLM CONFIGURATION
# =========================================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=30,
    max_tokens=3000
)


# =========================================================
# 2. EXPERIENCE-LEVEL SCORING RUBRICS
# =========================================================

# ---------------------------------------------------------
# GENERAL RESUME SCORING
# ---------------------------------------------------------
# Evaluates overall resume strength independently of a
# particular target job or role.
#
# Dimensions:
# - Technical skills
# - Projects
# - Experience
# - Education and certifications
# - Resume quality
# ---------------------------------------------------------

GENERAL_SCORING_RUBRICS: Dict[str, Dict[str, float]] = {

    "Fresher": {
        "technical_skills": 0.25,
        "projects": 0.30,
        "experience": 0.10,
        "education_certifications": 0.15,
        "resume_quality": 0.20
    },

    "Junior": {
        "technical_skills": 0.25,
        "projects": 0.20,
        "experience": 0.25,
        "education_certifications": 0.10,
        "resume_quality": 0.20
    },

    "Experienced": {
        "technical_skills": 0.20,
        "projects": 0.15,
        "experience": 0.35,
        "education_certifications": 0.10,
        "resume_quality": 0.20
    }
}


# ---------------------------------------------------------
# ROLE-SPECIFIC RESUME SCORING
# ---------------------------------------------------------
# Evaluates the candidate against a particular target role.
#
# Dimensions:
# - Technical skills
# - Projects
# - Experience
# - Role knowledge
# - Education and certifications
# ---------------------------------------------------------

ROLE_SCORING_RUBRICS: Dict[str, Dict[str, float]] = {

    "Fresher": {
        "technical_skills": 0.25,
        "projects": 0.50,
        "experience": 0.05,
        "role_knowledge": 0.15,
        "education_certifications": 0.05
    },

    "Junior": {
        "technical_skills": 0.25,
        "projects": 0.30,
        "experience": 0.25,
        "role_knowledge": 0.15,
        "education_certifications": 0.05
    },

    "Experienced": {
        "technical_skills": 0.20,
        "projects": 0.20,
        "experience": 0.40,
        "role_knowledge": 0.15,
        "education_certifications": 0.05
    }
}


def normalize_candidate_level(candidate_level: str) -> str:
    """Normalize the candidate's experience level."""

    if not candidate_level:
        raise ValueError(
            "Candidate experience level is missing."
        )

    value = candidate_level.strip().lower()

    aliases = {

        "fresher": "Fresher",
        "entry-level": "Fresher",
        "entry level": "Fresher",
        "graduate": "Fresher",

        "junior": "Junior",
        "early-career": "Junior",
        "early career": "Junior",

        "experienced": "Experienced",
        "mid-level": "Experienced",
        "mid level": "Experienced",
        "senior": "Experienced"
    }

    if value not in aliases:
        raise ValueError(
            f"Unsupported candidate level: {candidate_level!r}"
        )

    return aliases[value]


def validate_weights(
    weights: Dict[str, float],
    level: str,
    rubric_name: str
) -> Dict[str, float]:
    """Validate that scoring weights sum to 100%."""

    if not abs(sum(weights.values()) - 1.0) < 1e-9:

        raise ValueError(
            f"{rubric_name} weights for {level} "
            "do not sum to 1."
        )

    return weights


def get_general_scoring_weights(
    candidate_level: str
) -> Dict[str, float]:
    """Get experience-level weights for general scoring."""

    normalized_level = normalize_candidate_level(
        candidate_level
    )

    weights = GENERAL_SCORING_RUBRICS[
        normalized_level
    ].copy()

    return validate_weights(
        weights,
        normalized_level,
        "General scoring"
    )


def get_role_scoring_weights(
    candidate_level: str
) -> Dict[str, float]:
    """Get experience-level weights for role-specific scoring."""

    normalized_level = normalize_candidate_level(
        candidate_level
    )

    weights = ROLE_SCORING_RUBRICS[
        normalized_level
    ].copy()

    return validate_weights(
        weights,
        normalized_level,
        "Role-specific scoring"
    )


# =========================================================
# 3. PYDANTIC DIMENSION NORMALIZATION
# =========================================================

def normalize_dimension_payload(
    value,
    expected_fields
):
    """
    Normalize supported dimension score formats.

    Supported formats:

    1. Dictionary of numeric scores.
    2. Dictionary containing nested score values.
    3. List of dimension dictionaries.
    4. List of numeric scores.

    Returns a dictionary containing the exact expected keys.
    """

    # -----------------------------------------------------
    # FORMAT 1: DICTIONARY
    # -----------------------------------------------------

    if isinstance(value, dict):

        normalized = {}

        aliases = {
            "education_certifications": [
                "education_certifications",
                "education",
                "certifications"
            ]
        }

        for field in expected_fields:

            possible_keys = aliases.get(
                field,
                [field]
            )

            field_value = None

            for key in possible_keys:

                if key in value:

                    field_value = value[key]
                    break

            # Support nested score dictionaries.
            if isinstance(field_value, dict):

                field_value = field_value.get("score")

            if field_value is None:

                raise ValueError(
                    f"Missing score for {field}"
                )

            normalized[field] = field_value

        return normalized

    # -----------------------------------------------------
    # FORMAT 2: LIST
    # -----------------------------------------------------

    if isinstance(value, (list, tuple)):

        if len(value) != len(expected_fields):

            raise ValueError(
                f"Exactly {len(expected_fields)} "
                "dimension scores are required."
            )

        # -------------------------------------------------
        # LIST OF DIMENSION DICTIONARIES
        # -------------------------------------------------

        if all(
            isinstance(item, dict)
            for item in value
        ):

            normalized = {}

            aliases = {
                "education": "education_certifications",
                "certifications": "education_certifications"
            }

            for item in value:

                dimension = item.get("dimension")

                dimension = aliases.get(
                    dimension,
                    dimension
                )

                score = item.get("score")

                if dimension not in expected_fields:

                    raise ValueError(
                        f"Unknown dimension: {dimension}"
                    )

                if score is None:

                    raise ValueError(
                        f"Missing score for {dimension}"
                    )

                if dimension in normalized:

                    raise ValueError(
                        f"Duplicate dimension: {dimension}"
                    )

                normalized[dimension] = score

            missing = (
                set(expected_fields)
                - set(normalized)
            )

            if missing:

                raise ValueError(
                    f"Missing dimensions: {missing}"
                )

            return normalized

        # -------------------------------------------------
        # LIST OF NUMERIC SCORES
        # -------------------------------------------------

        if all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            for item in value
        ):

            return dict(
                zip(expected_fields, value)
            )

    raise ValueError(
        "Unsupported dimension_scores format."
    )


# =========================================================
# 4. PYDANTIC OUTPUT SCHEMAS
# =========================================================

# ---------------------------------------------------------
# GENERAL RESUME SCORING SCHEMA
# ---------------------------------------------------------

class GeneralDimensionScores(BaseModel):

    technical_skills: float = Field(ge=0, le=10)

    projects: float = Field(ge=0, le=10)

    experience: float = Field(ge=0, le=10)

    education_certifications: float = Field(
        ge=0,
        le=10
    )

    resume_quality: float = Field(ge=0, le=10)


class GeneralResumeScore(BaseModel):

    candidate_role: str

    experience_level: str

    dimension_scores: GeneralDimensionScores

    justification: str

    evidence: list[str]

    strengths: list[str]

    skill_gaps: list[str]

    improvements: list[str]

    red_flags: list[str]

    reasoning: str

    overall_score: float = 0.0

    recommendation_tier: str = "Pending"

    @field_validator(
        "dimension_scores",
        mode="before"
    )
    @classmethod
    def normalize_general_dimensions(cls, value):

        expected_fields = [

            "technical_skills",

            "projects",

            "experience",

            "education_certifications",

            "resume_quality"

        ]

        return normalize_dimension_payload(
            value,
            expected_fields
        )


# ---------------------------------------------------------
# ROLE-SPECIFIC RESUME SCORING SCHEMA
# ---------------------------------------------------------

class DimensionScores(BaseModel):

    technical_skills: float = Field(ge=0, le=10)

    projects: float = Field(ge=0, le=10)

    experience: float = Field(ge=0, le=10)

    role_knowledge: float = Field(ge=0, le=10)

    education_certifications: float = Field(
        ge=0,
        le=10
    )


class ResumeScore(BaseModel):

    candidate_role: str

    experience_level: str

    dimension_scores: DimensionScores

    justification: str

    evidence: list[str]

    strengths: list[str]

    skill_gaps: list[str]

    improvements: list[str]

    red_flags: list[str]

    reasoning: str

    overall_score: float = 0.0

    recommendation_tier: str = "Pending"

    @field_validator(
        "dimension_scores",
        mode="before"
    )
    @classmethod
    def normalize_role_dimensions(cls, value):

        expected_fields = [

            "technical_skills",

            "projects",

            "experience",

            "role_knowledge",

            "education_certifications"

        ]

        return normalize_dimension_payload(
            value,
            expected_fields
        )


# =========================================================
# 5. STRUCTURED LLM OUTPUTS
# =========================================================

general_scoring_llm = model.with_structured_output(
    GeneralResumeScore,
    method="json_mode"
)


scoring_llm = model.with_structured_output(
    ResumeScore,
    method="json_mode"
)


# =========================================================
# 6. PYTHON SCORING FUNCTIONS
# =========================================================

def calculate_final_score(
    dimension_scores,
    weights: Dict[str, float]
) -> float:
    """
    Calculate the weighted score.

    All dimension scores are numeric.
    Python exclusively calculates the final score.
    """

    raw_total = 0.0

    for criterion, weight in weights.items():

        score = getattr(
            dimension_scores,
            criterion
        )

        raw_total += score * weight

    return raw_total


# ---------------------------------------------------------
# ROLE-SPECIFIC RECOMMENDATION
# ---------------------------------------------------------

def get_role_recommendation_tier(
    final_score: float
) -> str:

    if final_score >= 8.0:
        return "Strong Fit"

    if final_score >= 6.5:
        return "Good Fit"

    if final_score >= 5.0:
        return "Partial Fit"

    return "Weak Fit"


# ---------------------------------------------------------
# GENERAL RESUME RECOMMENDATION
# ---------------------------------------------------------

def get_general_recommendation_tier(
    final_score: float
) -> str:

    if final_score >= 8.0:
        return "Excellent Resume"

    if final_score >= 6.5:
        return "Strong Resume"

    if final_score >= 5.0:
        return "Moderate Resume"

    return "Needs Improvement"


# ---------------------------------------------------------
# GENERAL SCORE FINALIZATION
# ---------------------------------------------------------

def finalize_general_score(
    result: GeneralResumeScore,
    candidate_level: str
) -> GeneralResumeScore:
    """
    Calculate the general resume score using
    the candidate's experience-level weights.
    """

    normalized_level = normalize_candidate_level(
        candidate_level
    )

    weights = get_general_scoring_weights(
        normalized_level
    )

    result.experience_level = normalized_level

    raw_score = calculate_final_score(
        result.dimension_scores,
        weights
    )

    result.overall_score = round(
        raw_score,
        1
    )

    result.recommendation_tier = (
        get_general_recommendation_tier(
            raw_score
        )
    )

    return result


# ---------------------------------------------------------
# ROLE-SPECIFIC SCORE FINALIZATION
# ---------------------------------------------------------

def finalize_role_score(
    result: ResumeScore,
    candidate_level: str
) -> ResumeScore:
    """
    Calculate the role-specific score using
    the candidate's experience-level weights.
    """

    normalized_level = normalize_candidate_level(
        candidate_level
    )

    weights = get_role_scoring_weights(
        normalized_level
    )

    result.experience_level = normalized_level

    raw_score = calculate_final_score(
        result.dimension_scores,
        weights
    )

    result.overall_score = round(
        raw_score,
        1
    )

    result.recommendation_tier = (
        get_role_recommendation_tier(
            raw_score
        )
    )

    return result


# =========================================================
# 7. WEIGHT FORMATTING FUNCTIONS
# =========================================================

def get_general_weights_text(
    weights: Dict[str, float]
) -> str:
    """Format general scoring weights."""

    labels = {

        "technical_skills":
            "Technical skills and demonstrated ability",

        "projects":
            "Projects and practical implementation",

        "experience":
            "Professional experience and practical exposure",

        "education_certifications":
            "Education and certifications",

        "resume_quality":
            "Resume quality, clarity and evidence"

    }

    return "\n".join(

        f"- {labels[key]}: {value:.0%}"

        for key, value in weights.items()

    )


def get_role_weights_text(
    weights: Dict[str, float]
) -> str:
    """Format role-specific scoring weights."""

    labels = {

        "technical_skills":
            "Technical skills relevant to target role",

        "projects":
            "Projects and practical implementation relevant to target role",

        "experience":
            "Relevant professional experience and practical exposure",

        "role_knowledge":
            "Target-role knowledge",

        "education_certifications":
            "Education and certifications"

    }

    return "\n".join(

        f"- {labels[key]}: {value:.0%}"

        for key, value in weights.items()

    )


# =========================================================
# 8. GENERAL RESUME EVALUATION INSTRUCTIONS
# =========================================================

GENERAL_EVALUATION_INSTRUCTIONS = """

You are an expert resume evaluator.

Your task is to evaluate the overall strength and quality
of a candidate's resume.

This is a GENERAL RESUME EVALUATION.

It is NOT a job matching evaluation.

Do not assume the candidate is applying for a particular
job, company, industry or target position.

Evaluate the resume independently of any target role.

=========================================================
EXPERIENCE-LEVEL CONTEXT
=========================================================

The candidate's experience level and scoring weights are
provided separately.

Use the supplied experience-level weights.

Do not change, invent or calculate different weights.

The weights determine the relative importance of the
evaluation dimensions.

They do not directly determine the dimension scores.

=========================================================
SCORING DIMENSIONS
=========================================================

Evaluate exactly five dimensions.

1. TECHNICAL SKILLS

Evaluate:

- Breadth of technical skills.
- Depth of demonstrated knowledge.
- Programming languages.
- Frameworks and libraries.
- Tools and technologies.
- Practical proficiency.
- Evidence of applying technical skills.
- Technical problem-solving.

Distinguish between skills merely listed and skills
demonstrated through actual work.

Do not judge technical skills against a particular
target job.

2. PROJECTS

Evaluate:

- Implementation depth.
- Technical complexity.
- Problem-solving.
- Candidate ownership.
- Architecture and design decisions.
- Testing and evaluation.
- Practical implementation.
- Demonstrated outcomes.

Evaluate the quality of projects, not their relevance
to a particular target job.

Give credit to genuine implementations.

Do not require production deployment or advanced metrics
from every candidate.

3. EXPERIENCE

Evaluate:

- Professional responsibilities.
- Practical technical work.
- Internships.
- Contributions and ownership.
- Duration and progression.
- Demonstrated impact.
- Evidence of applying professional skills.

Distinguish between:

- Full-time professional experience.
- Internships.
- Academic projects.
- Personal projects.

For Freshers:

- Do not treat the absence of full-time employment
  as a failure.
- Evaluate internships and practical exposure honestly.
- Consider the responsibilities actually performed.
- Projects must be assessed separately under the projects
  dimension.
- Do not double-count projects as professional employment.

For Junior candidates:

- Consider both practical projects and early professional
  contributions.
- Evaluate the depth of actual responsibilities.

For Experienced candidates:

- Examine professional experience in greater depth.
- Consider duration, technical ownership, responsibilities,
  progression and measurable outcomes.

4. EDUCATION AND CERTIFICATIONS

Evaluate:

- Academic qualifications.
- Educational foundations.
- Relevant coursework.
- Certifications.
- Structured technical learning.
- Evidence of continued learning.

Do not automatically penalize unrelated degrees.

Do not automatically penalize missing optional certifications.

Evaluate the actual educational background rather than
assuming a particular specialization.

5. RESUME QUALITY

Evaluate:

- Organization and structure.
- Clarity and readability.
- Completeness of important sections.
- Specificity of descriptions.
- Use of measurable achievements where appropriate.
- Consistency.
- Professional presentation.
- Whether claims are supported with concrete evidence.
- Whether responsibilities and accomplishments are
  communicated clearly.

Do not confuse visual decoration with resume quality.

Do not penalize a candidate simply for lacking a particular
resume template or design.

=========================================================
GENERAL SCORING SCALE
=========================================================

Use the following 0–10 scale independently for every dimension.

0–2:
Very limited evidence or extremely weak presentation.

3–4:
Basic foundations, superficial evidence or significant
weaknesses.

5–6:
Reasonable foundations and genuine evidence, but limited
depth or consistency.

7:
Strong evidence and good overall resume quality.

8:
Very strong evidence, depth, clarity and achievements.

9–10:
Exceptional evidence, substantial depth, ownership and
professional presentation.

Do not inflate scores.

A score of 8 or above requires strong supporting evidence.

=========================================================
EVIDENCE AND FAIRNESS RULES
=========================================================

- Evaluate the resume independently of any target role.
- Do not use a role guide or job description.
- Do not penalize the candidate for lacking skills in a
  particular industry or specialization.
- Do not assume missing information proves the candidate
  lacks the underlying ability.
- Distinguish missing evidence from demonstrated weakness.
- Evaluate the quality of evidence rather than simply
  counting listed skills.
- Never invent achievements, responsibilities, technologies
  or metrics.
- Do not double-count the same evidence without justification.
- Evaluate projects independently from professional experience.
- Do not reward irrelevant certifications merely for
  their quantity.
- Do not infer personal attributes.
- Consider the candidate's experience level when evaluating
  the depth of their resume.

=========================================================
OUTPUT CONTENT REQUIREMENTS
=========================================================

Return:

1. candidate_role:

   Use exactly:
   "General Resume Evaluation"

2. experience_level:

   Use the supplied candidate experience level.

3. dimension_scores:

   Return exactly five numeric scores between 0 and 10:

   - technical_skills
   - projects
   - experience
   - education_certifications
   - resume_quality

4. justification:

   A concise explanation of the overall resume evaluation.

5. evidence:

   Short statements directly supported by resume evidence.

   If no supporting evidence exists, return an empty list.

6. strengths:

   Provide 2–5 meaningful, evidence-based strengths
   where possible.

7. skill_gaps:

   Identify weaknesses in demonstrated skills or missing
   supporting evidence.

   Do not describe gaps against a target job.

8. improvements:

   Provide 2–4 actionable improvements to strengthen
   the resume and its evidence.

9. red_flags:

   Include only genuine concerns supported by resume evidence.

   If none exist, return an empty list.

10. reasoning:

    Provide 3–5 sentences explaining the evaluation.

    Do not state or calculate an overall numeric score
    or recommendation tier.

=========================================================
IMPORTANT PYTHON-OWNED FIELDS
=========================================================

Python exclusively calculates:

- Dimension weights.
- Overall weighted score.
- Recommendation tier.

Do not calculate, estimate or return your own overall score.

Do not add dimension weights or contributions.

The overall_score and recommendation_tier fields in the
output example are placeholders.

Python replaces them after your evaluation.

=========================================================
MANDATORY OUTPUT FORMAT
=========================================================

Return a JSON object matching this structure.

This is a FORMAT EXAMPLE, not a scoring reference.

{{
    "candidate_role": "General Resume Evaluation",

    "experience_level": "Fresher",

    "dimension_scores": {{
        "technical_skills": 6.0,
        "projects": 7.0,
        "experience": 4.0,
        "education_certifications": 7.0,
        "resume_quality": 6.0
    }},

    "justification": "The resume demonstrates foundational
    technical skills and practical project work, with limited
    professional experience.",

    "evidence": [
        "Demonstrated Python programming through a project."
    ],

    "strengths": [
        "Practical technical implementation."
    ],

    "skill_gaps": [
        "Limited evidence of advanced technical proficiency."
    ],

    "improvements": [
        "Add measurable outcomes to project descriptions."
    ],

    "red_flags": [],

    "reasoning": "The resume demonstrates foundational
    technical ability and some practical implementation.
    Professional experience is limited, which is considered
    in the context of the candidate's experience level.
    Clearer achievements and stronger evidence would improve
    the overall resume.",

    "overall_score": 0.0,

    "recommendation_tier": "Pending"
}}

STRICT FORMAT RULES:

- Return dimension_scores as a JSON OBJECT.
- Do not return dimension_scores as a list.
- Use exactly the five specified dimension keys.
- Every dimension score must be numeric, not a string.
- Do not wrap individual dimension scores in objects.
- Do not rename any fields.
- Do not omit any required field.
- Evidence, strengths, skill_gaps, improvements and red_flags
  must always be lists of strings.
- Return only structured JSON matching the example and schema.

The exact dimension_scores keys must be:

- technical_skills
- projects
- experience
- education_certifications
- resume_quality

Do not use role_knowledge in general evaluation.

"""


# =========================================================
# 9. GENERAL SCORING PROMPT AND CHAIN
# =========================================================

general_scoring_prompt = ChatPromptTemplate([

    ("system", GENERAL_EVALUATION_INSTRUCTIONS),

    ("human", """
Candidate experience level:
{candidate_level}

General scoring weights:
{weights_text}

Candidate resume JSON:
{candidate_resume_json}
""")

])


general_scoring_chain = (
    general_scoring_prompt | general_scoring_llm
)


# =========================================================
# 10. GENERAL RESUME SCORING GENERATOR
# =========================================================

def generate_general(
    file_path,
    return_context=False
):
    """
    Generate a general resume evaluation.

    This evaluation is independent of any target role.

    It does not use:
    - Role guides.
    - Benchmark resumes.
    - Retrieval.
    """

    # -----------------------------------------------------
    # STEP 1: EXTRACT RESUME
    # -----------------------------------------------------

    candidate_data = load_file(file_path)

    resume_data = candidate_data["resume_json"]

    experience_data = candidate_data.get(
        "experience_data",
        {}
    )

    candidate_level = normalize_candidate_level(
        experience_data.get("candidate_level")
    )

    weights = get_general_scoring_weights(
        candidate_level
    )

    candidate_resume_json = json.dumps(
        resume_data,
        ensure_ascii=False
    )

    # -----------------------------------------------------
    # STEP 2: GENERATE GENERAL EVALUATION
    # -----------------------------------------------------

    result = general_scoring_chain.invoke({

        "candidate_level": candidate_level,

        "weights_text": get_general_weights_text(
            weights
        ),

        "candidate_resume_json": candidate_resume_json

    })

    # -----------------------------------------------------
    # STEP 3: CALCULATE FINAL SCORE IN PYTHON
    # -----------------------------------------------------

    result = finalize_general_score(
        result,
        candidate_level
    )

    # -----------------------------------------------------
    # STEP 4: DISPLAY RESULT
    # -----------------------------------------------------

    print(
        "\n========== GENERAL RESUME SCORE =========="
    )

    print(
        result.model_dump_json(indent=4)
    )

    if return_context:

        scoring_context = [

            "Candidate Resume:\n"
            + candidate_resume_json,

            "Experience Level:\n"
            + candidate_level,

            "General Scoring Weights:\n"
            + json.dumps(
                weights,
                indent=2
            )

        ]

        return result, scoring_context

    return result


# =========================================================
# 11. ROLE-SPECIFIC EVALUATION INSTRUCTIONS
# =========================================================

ROLE_SPECIFIC_EVALUATION_INSTRUCTIONS = """

You are an expert technical resume evaluator.

Your task is to evaluate a candidate against a supplied
target role and role guide.

This is a ROLE-SPECIFIC RESUME EVALUATION.

The candidate's resume must be evaluated against the
requirements and responsibilities of the target role.

=========================================================
EXPERIENCE-LEVEL CONTEXT
=========================================================

The candidate's experience level and scoring weights are
provided separately.

The scoring weights differ depending on the candidate's level.

Apply the supplied weights when prioritizing the evaluation.

Do not change, invent or calculate different weights.

=========================================================
SCORING DIMENSIONS
=========================================================

Evaluate exactly five dimensions.

1. TECHNICAL SKILLS

Evaluate:

- Programming languages.
- Frameworks and libraries.
- Relevant tools and technologies.
- Practical proficiency.
- Evidence of applying technical skills.
- Relevance to the target role.

Distinguish between skills merely listed and skills
demonstrated through actual work.

2. PROJECTS

Evaluate:

- Relevance to the target role.
- Implementation depth.
- Technical complexity.
- Problem-solving.
- Candidate ownership.
- Architecture and design decisions.
- Testing and evaluation.
- Demonstrated outcomes.

Give credit to genuine implementations.

Do not require production deployment or advanced metrics
from every candidate.

3. EXPERIENCE

Evaluate:

- Relevant professional responsibilities.
- Practical technical work.
- Ownership and contributions.
- Duration and relevance.
- Demonstrated impact.

Distinguish between:

- Full-time professional experience.
- Internships.
- Academic projects.
- Personal projects.

For Freshers:

- Do not treat the absence of full-time employment as a failure.
- Evaluate internships and practical exposure honestly.
- Projects must be assessed separately under the projects
  dimension.
- Do not double-count projects as professional employment.

For Junior candidates:

- Consider both practical projects and early professional
  contributions.
- Evaluate the depth of their actual responsibilities.

For Experienced candidates:

- Examine professional experience in greater depth.
- Consider duration, technical ownership, responsibilities,
  relevance and measurable outcomes.

4. ROLE KNOWLEDGE

Evaluate:

- Understanding of core concepts.
- Knowledge of relevant workflows.
- Understanding of technical decisions.
- Familiarity with role-specific responsibilities.
- Evidence of applying relevant concepts.

Do not assume expertise merely because a technology appears
in the resume.

5. EDUCATION AND CERTIFICATIONS

Evaluate:

- Relevant academic foundations.
- Relevant degrees and coursework.
- Relevant certifications.
- Structured technical learning.

Do not automatically penalize unrelated degrees or missing
optional certifications.

=========================================================
GENERAL SCORING SCALE
=========================================================

Use the following 0–10 scale independently for every dimension.

0–2:
Little or no relevant evidence.

3–4:
Weak foundational knowledge or superficial exposure.

5–6:
Relevant foundational ability and genuine practical work,
but limited depth.

7:
Competent candidate meeting core expectations.

8:
Strong candidate demonstrating substantial implementation depth.

9–10:
Exceptional technical depth, ownership and results.

Do not inflate scores.

A score of 8 or above requires strong supporting evidence.

=========================================================
EVIDENCE AND FAIRNESS RULES
=========================================================

- Evaluate each dimension independently.
- Use only information supported by the resume.
- Never invent skills, experience, achievements, technologies,
  responsibilities, project details or metrics.
- Do not confuse a listed skill with demonstrated proficiency.
- Do not double-count the same evidence without a clear reason.
- Missing evidence is not automatically proof that the candidate
  lacks a skill.
- Clearly distinguish missing evidence from demonstrated weakness.
- Do not penalize candidates for demographic or personal attributes.
- Use the role guide to understand the target position.
- Projects and internships can demonstrate practical ability.
- Do not require every role-guide item from an entry-level candidate.

=========================================================
OUTPUT CONTENT REQUIREMENTS
=========================================================

Return:

1. candidate_role:

   The target role being evaluated.

2. experience_level:

   The supplied candidate experience level.

3. dimension_scores:

   Exactly five numeric scores, each between 0 and 10.

4. justification:

   A concise explanation of the overall dimension-level
   evaluation. Do not calculate the overall score.

5. evidence:

   A list of short, resume-grounded evidence statements.

   If no supporting evidence exists, return an empty list.

6. strengths:

   Provide 2–5 meaningful, evidence-based strengths where possible.

7. skill_gaps:

   List meaningful role-related gaps supported by the evaluation.

   Do not invent missing skills.

8. improvements:

   Provide 2–4 actionable improvements.

9. red_flags:

   Include only genuine concerns supported by resume evidence.

   If none exist, return an empty list.

10. reasoning:

    Provide 3–5 sentences explaining the evaluation.

    Do not state or calculate an overall numeric score or
    recommendation tier.

=========================================================
IMPORTANT PYTHON-OWNED FIELDS
=========================================================

Python exclusively calculates:

- Dimension weights.
- Overall weighted score.
- Recommendation tier.

Do not calculate, estimate or return your own overall score.

Do not add dimension weights or contributions.

The overall_score and recommendation_tier fields in the
output example are placeholders.

Python replaces them after your evaluation.

=========================================================
MANDATORY OUTPUT FORMAT
=========================================================

The following is an example of the exact JSON structure
you must return.

This is a FORMAT EXAMPLE, not a scoring reference.

{{
    "candidate_role": "AI Engineer",

    "experience_level": "Fresher",

    "dimension_scores": {{
        "technical_skills": 7.0,
        "projects": 8.0,
        "experience": 5.0,
        "role_knowledge": 7.0,
        "education_certifications": 6.0
    }},

    "justification": "The candidate demonstrates relevant
    technical skills and practical project implementation.
    Professional experience is limited, but the candidate
    has some practical exposure through internships.",

    "evidence": [
        "Implemented a retrieval-augmented generation project.",
        "Used Python and LangChain in practical projects.",
        "Completed a relevant technical internship."
    ],

    "strengths": [
        "Demonstrated practical Python implementation.",
        "Relevant project experience.",
        "Exposure to LLM application development."
    ],

    "skill_gaps": [
        "Limited evidence of production deployment.",
        "Limited evidence of large-scale system development."
    ],

    "improvements": [
        "Add measurable evaluation results to projects.",
        "Demonstrate deployment and monitoring experience."
    ],

    "red_flags": [],

    "reasoning": "The candidate demonstrates relevant technical
    foundations and practical implementation through projects.
    The projects provide evidence of applying technologies rather
    than merely listing them. Professional experience is limited,
    which is consistent with the supplied fresher classification.
    Further evidence of deployment and system-level ownership
    would strengthen the application.",

    "overall_score": 0.0,

    "recommendation_tier": "Pending"
}}

STRICT FORMAT RULES:

- Return dimension_scores as a JSON OBJECT.
- Do not return dimension_scores as a list.
- Every dimension score must be numeric, not a string.
- Do not wrap individual dimension scores in objects.
- Do not rename any fields.
- Do not omit any required field.
- Evidence, strengths, skill_gaps, improvements and red_flags
  must always be lists of strings.
- Return only structured JSON matching the example and schema.

The exact dimension_scores keys must be:

- technical_skills
- projects
- experience
- role_knowledge
- education_certifications

Never use shortened alternatives such as:

- education
- certifications
- academic_background

Use education_certifications exactly.

"""


# =========================================================
# 12. ROLE-SPECIFIC SCORING PROMPT AND CHAIN
# =========================================================

role_specific_system_prompt = (
    ROLE_SPECIFIC_EVALUATION_INSTRUCTIONS
    + """

ROLE-SPECIFIC EVALUATION MODE:

You are also provided with retrieved benchmark resumes.

Benchmark records may contain:

- resume
- human_score
- score_reason
- quality_label
- rrf_score
- category_evaluations
- evaluation_status

Use these records only as calibration references.

=========================================================
BENCHMARK EVALUATION RULES
=========================================================

1. Compare actual implementation evidence, project depth,
   responsibilities, ownership and outcomes.

2. Use human scores and score reasons to understand the
   evaluation standard, not as scores to copy.

3. Never average benchmark scores to calculate the candidate score.

4. Never treat RRF scores as candidate quality scores.

5. A higher RRF score does not mean a stronger candidate.

6. Similar project titles or resume structures do not establish
   equivalent technical ability.

7. Do not invent benchmark evidence.

8. Treat draft or unreviewed benchmark evaluations cautiously.

9. Evaluate the candidate independently against the role guide
   and the supplied experience-level scoring weights.

10. Do not penalize a candidate merely because their background
    differs from retrieved benchmarks.

11. Do not assume that the strongest retrieved benchmark is
    automatically the standard the candidate must meet.

12. Use benchmarks to improve consistency, not to replace
    independent evaluation.

"""
)


role_specific_scoring_prompt = ChatPromptTemplate([

    ("system", role_specific_system_prompt),

    ("human", """
Target role:
{role_name}

Candidate experience level:
{candidate_level}

Scoring weights for this candidate:
{weights_text}

Role guide:
{role_guide}

Candidate resume JSON:
{candidate_resume_json}

Retrieved benchmark resumes:
{benchmark_resume_json}
""")

])


role_specific_scoring_chain = (
    role_specific_scoring_prompt | scoring_llm
)


# =========================================================
# 13. ROLE-SPECIFIC RESUME SCORING GENERATOR
# =========================================================

def generate_role_specific(
    file_path,
    role_name,
    return_context=False,
    k=1
):
    """
    Generate a role-specific resume evaluation.

    This evaluation uses:

    - Candidate resume extraction.
    - Experience-level filtered benchmark retrieval.
    - Role guide.
    - Benchmark human scores and score reasons.
    - Role-specific scoring weights.
    """

    # -----------------------------------------------------
    # STEP 1: RETRIEVE BENCHMARKS
    # -----------------------------------------------------

    docs, candidate_resume_json = retrieve_role_level(

        file_path=file_path,

        role_name=role_name,

        k=k

    )

    if not docs:

        raise ValueError(
            "No benchmark resumes were retrieved for "
            f"{role_name!r} and the candidate's experience level."
        )

    # -----------------------------------------------------
    # STEP 2: DETERMINE CANDIDATE EXPERIENCE LEVEL
    # -----------------------------------------------------

    # Retriever filters benchmarks by candidate level.
    # All retrieved documents should belong to the same level.

    candidate_level = normalize_candidate_level(
        docs[0]["candidate_level"]
    )

    # Verify that retrieved documents have a consistent level.

    for doc in docs:

        doc_level = normalize_candidate_level(
            doc["candidate_level"]
        )

        if doc_level != candidate_level:

            raise ValueError(
                "Retrieved benchmarks contain inconsistent "
                "candidate experience levels."
            )

    weights = get_role_scoring_weights(
        candidate_level
    )

    # -----------------------------------------------------
    # STEP 3: PREPARE BENCHMARK DATA
    # -----------------------------------------------------

    benchmark_resumes = []

    print(
        "\n========== RETRIEVED BENCHMARKS =========="
    )

    for rank, doc in enumerate(
        docs,
        start=1
    ):

        benchmark_resumes.append({

            "rank": rank,

            "resume_id": doc["resume_id"],

            "candidate_level": doc["candidate_level"],

            "role_name": doc["role_name"],

            "rrf_score": doc["rrf_score"],

            "resume": doc["resume"],

            "human_score": doc["human_score"],

            "score_reason": doc["score_reason"],

            "quality_label": doc["quality_label"],

            "category_evaluations": doc["category_evaluations"],

            "evaluation_status": doc["evaluation_status"]

        })

        print(f"\nBenchmark Rank: {rank}")

        print(
            "Name:",
            doc.get("candidate_name")
        )

        print(
            "Resume ID:",
            doc["resume_id"]
        )

        print(
            "Human Score:",
            doc["human_score"]
        )

        print(
            "Quality:",
            doc["quality_label"]
        )

        print(
            "RRF Score:",
            round(doc["rrf_score"], 6)
        )

        print(
            "Evaluation Status:",
            doc["evaluation_status"]
        )

    # -----------------------------------------------------
    # STEP 4: GET ROLE GUIDE
    # -----------------------------------------------------

    role_guide = get_role_guide(
        role_name=role_name
    )

    if not role_guide:

        raise ValueError(
            f"No role guide found for {role_name!r}. "
            "Add the role guide to the role_guides table first."
        )

    # -----------------------------------------------------
    # STEP 5: SERIALIZE BENCHMARK DATA
    # -----------------------------------------------------

    benchmark_resume_json = json.dumps(

        benchmark_resumes,

        indent=2,

        ensure_ascii=False

    )

    # -----------------------------------------------------
    # STEP 6: GENERATE ROLE-SPECIFIC EVALUATION
    # -----------------------------------------------------

    result = role_specific_scoring_chain.invoke({

        "role_name": role_name,

        "candidate_level": candidate_level,

        "weights_text": get_role_weights_text(
            weights
        ),

        "role_guide": role_guide,

        "candidate_resume_json": candidate_resume_json,

        "benchmark_resume_json": benchmark_resume_json

    })

    # -----------------------------------------------------
    # STEP 7: CALCULATE FINAL SCORE IN PYTHON
    # -----------------------------------------------------

    result = finalize_role_score(
        result,
        candidate_level
    )

    # -----------------------------------------------------
    # STEP 8: DISPLAY RESULT
    # -----------------------------------------------------

    print(
        "\n========== ROLE-SPECIFIC RESUME SCORE =========="
    )

    print(
        result.model_dump_json(indent=4)
    )

    if return_context:

        scoring_context = [

            "Candidate Resume:\n"
            + candidate_resume_json,

            "Role Guide:\n"
            + str(role_guide),

            "Experience Level:\n"
            + candidate_level,

            "Scoring Weights:\n"
            + json.dumps(
                weights,
                indent=2
            ),

            "Retrieved Benchmark Resumes:\n"
            + benchmark_resume_json

        ]

        return result, scoring_context

    return result


# =========================================================
# 14. MANUAL TEST ENTRY POINT
# =========================================================

if __name__ == "__main__":

    file_path = "dummy_resumes/strong_07.pdf"

    role_name = "AI Engineer"

    # -----------------------------------------------------
    # GENERAL EVALUATION
    # -----------------------------------------------------

    general_result = generate_general(

        file_path=file_path

    )

    print(
        "\nGENERAL SCORE RESULT"
    )

    print(
        general_result.model_dump_json(indent=4)
    )

    # print("waiting 20 seconds")
    # time.sleep(20)
    # -----------------------------------------------------
    # ROLE-SPECIFIC EVALUATION
    # -----------------------------------------------------

    # Uncomment when you want to test benchmark calibration.

    
    role_specific_result = generate_role_specific(
    
        file_path=file_path,
    
        role_name=role_name,
    
        k=2
    
    )
    
    print("\nROLE-SPECIFIC SCORE RESULT")
    
    print(
        role_specific_result.model_dump_json(indent=4)
    )
