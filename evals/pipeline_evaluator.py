import json
import csv
import time
from pathlib import Path
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)
from langchain_groq import ChatGroq
from deepeval.models import DeepEvalBaseLLM
from gen_eval_temp import (
    CustomFaithfulnessTemplate,
    CustomAnswerRelevancyTemplate,
    CustomContextualRelevancyTemplate
)
from generator_evaluator import GroqJudge
import ast
import json


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=120,
    max_tokens=4000,
    reasoning_effort="low"
)

judge = GroqJudge(model=llm)

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "evals" / "pipeline_outputs.json"

OUTPUT_FILE = PROJECT_DIR / "evals" / "pipeline_evaluation_results.csv"




faithfulness = FaithfulnessMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
    async_mode=False,
    penalize_ambiguous_claims=True,
    evaluation_template=CustomFaithfulnessTemplate
)

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
    async_mode=False,
    evaluation_template=CustomAnswerRelevancyTemplate
)

contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
    async_mode=False,
    evaluation_template=CustomContextualRelevancyTemplate
)



def load_pipeline_data():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Loaded {len(data)} evaluation records.")

    return data








def evaluate_pipeline(data):

    results = []

    for index, record in enumerate(data, start=1):

        print(f"\nEvaluating {index}/{len(data)}: {record['test_file']}")

        # Format existing retrieval context
        formatted_context = []

        for context in record["retrieval_context"]:

            label, content = context.split("\n", 1)

            try:
                parsed_content = ast.literal_eval(content)

                if isinstance(parsed_content, (dict, list)):
                    content = json.dumps(
                        parsed_content,
                        indent=2,
                        ensure_ascii=False
                    )

            except (ValueError, SyntaxError):
                pass

            formatted_context.append(f"{label}\n{content}")

        # Create test case
        test_case = LLMTestCase(
            input=record["input"],
            actual_output=record["actual_output"],
            retrieval_context=formatted_context
        )

        metrics = [
            faithfulness,
            answer_relevancy,
            contextual_relevancy
        ]
        

        result = {
            "test_file": record["test_file"],
            "human_score": record["human_score"]
        }

        for metric in metrics:

            metric.measure(test_case)

            result[metric.__class__.__name__] = metric.score
            result[f"{metric.__class__.__name__}_reason"] = metric.reason

            print(
                f"{metric.__class__.__name__}: "
                f"{metric.score}"
            )

            time.sleep(10)

        results.append(result)

        # Pause between resumes to reduce API rate-limit issues
        print("\nWaiting 60 seconds...")
        time.sleep(60)

    return results

def save_results(results):

    if not results:
        print("No evaluation results to save.")
        return

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_FILE}")


def calculate_averages(results):

    metric_names = [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "ContextualRelevancyMetric"
    ]

    
    print("\n========== AVERAGE PIPELINE METRICS ==========")

    for metric in metric_names:

        scores = [
            row[metric]
            for row in results
            if row.get(metric) is not None
        ]

        if scores:
            average = sum(scores) / len(scores)
            print(f"{metric}: {average:.4f}")


def pipeline_eval(test_files=None):

    # 1. Load saved pipeline outputs
    data = load_pipeline_data()

    # 2. Filter records to match the other evaluations
    if test_files is not None:

        test_files = set(test_files)

        data = [
            record
            for record in data
            if record["test_file"] in test_files
        ]

    # 3. Evaluate pipeline
    results = evaluate_pipeline(data)

    # 4. Calculate averages
    averages = calculate_averages(results)

    # 5. Save results
    save_results(results)

    return {
        "individual_results": results,
        "averages": averages
    }


def main():

    # Load previously exported pipeline results
    data = load_pipeline_data()
    # record = data[0]

    # print("File being read:", INPUT_FILE)
    # print("Number of contexts:", len(record["retrieval_context"]))

    # for i, context in enumerate(record["retrieval_context"], start=1):
    #     print(f"\nContext {i}:")
    #     print(context[:300])

    # Evaluate saved outputs (no retriever or generator calls)
    results = evaluate_pipeline(data)

    # Save individual evaluation results
    # save_results(results)

    # Calculate average metric scores
    calculate_averages(results)


if __name__ == "__main__":
    pipeline_eval()