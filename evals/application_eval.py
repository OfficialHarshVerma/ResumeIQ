
from pathlib import Path
import time
import json
import csv

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from generator_evaluator import GroqJudge
from langchain_groq import ChatGroq
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from user import generate


# ==========================================
# LLM JUDGE
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=120,
    max_tokens=4000,
    reasoning_effort="low"
)

judge = GroqJudge(model=llm)


# ==========================================
# 1. CORRECTNESS
# ==========================================

correctness = GEval(
    name="Correctness",
    criteria=(
        "Evaluate whether the application's final resume assessment "
        "is factually accurate and supported by the complete supplied "
        "context. Check scores, reasoning, strengths, skill gaps, and "
        "improvement suggestions against the candidate resume, role "
        "guide, and benchmark resumes. Penalize unsupported claims "
        "and contradictions."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT
    ],
    threshold=0.7,
    model=judge,
    async_mode=False
)


# ==========================================
# 2. COMPLETENESS
# ==========================================

completeness = GEval(
    name="Completeness",
    criteria=(
        "Evaluate whether the final resume assessment covers all "
        "important information expected from the application. Check "
        "whether it includes the overall score, all five criterion "
        "scores, reasoning, strengths, skill gaps, and actionable "
        "improvements. Penalize missing or inadequately explained "
        "sections."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT
    ],
    threshold=0.7,
    model=judge,
    async_mode=False
)


# ==========================================
# 3. TONE
# ==========================================

tone = GEval(
    name="Tone",
    criteria=(
        "Evaluate the tone of the final resume assessment. It should "
        "be professional, objective, respectful, constructive, and "
        "easy to understand. Penalize rude, discouraging, biased, "
        "overly casual, or unnecessarily complicated language."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    threshold=0.7,
    model=judge,
    async_mode=False
)



def evaluate_application(test_files):

    metrics = [
        correctness,
        completeness,
        tone
    ]

    evaluation_results = []

    for index, test_file in enumerate(test_files, start=1):

        print("\n" + "=" * 60)
        print(f"APPLICATION EVALUATION: {test_file}")
        print("=" * 60)

        # 1. Run the actual application
        result, retrieval_context = generate(
            file_path=test_file,
            role_name="AI Engineer",
            return_context=True
        )

        # 2. Convert structured output to JSON
        actual_output = result.model_dump_json(indent=2)

        # 3. Build evaluation context
        context = retrieval_context

        # 4. Create DeepEval test case
        test_case = LLMTestCase(
            input="Evaluate this candidate's suitability for the AI Engineer role.",
            actual_output=actual_output,
            context=context
        )

        # 5. Run all application metrics
        record = {
            "test_file": Path(test_file).name
        }

        for metric in metrics:

            metric.measure(test_case)

            record[metric.name] = metric.score
            record[f"{metric.name}_reason"] = metric.reason

            print(f"{metric.name}: {metric.score}")

            time.sleep(10)

        evaluation_results.append(record)

        if index < len(test_files):
            print("\nWaiting 60 seconds...")
            time.sleep(60)

    return evaluation_results


if __name__ == "__main__":

    test_folder = Path("evals/test_resumes/")

    test_files = [
        str(test_folder / f"{category}_{number:02d}.pdf")
        for category in ["strong","average","weak"]
        for number in range(8, 11)
    ]

    results = evaluate_application(test_files)

    print("\n========== APPLICATION EVALUATION ==========")

    for result in results:
        print(json.dumps(result, indent=2))