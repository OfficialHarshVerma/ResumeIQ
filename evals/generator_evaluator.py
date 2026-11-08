
import json
from pathlib import Path
import time
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from gen_eval_temp import CustomFaithfulnessTemplate,CustomAnswerRelevancyTemplate
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


# 1. Define paths
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent

GENERATED_FILE = EVAL_DIR / "generated_outputs.json"

load_dotenv(PROJECT_DIR / ".env")


# 2. Initialize Groq judge model
judge_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=120,
    max_tokens=10000,
    reasoning_effort="low"
)

# 3. Create custom DeepEval judge


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def get_model_name(self):
        return "Groq GPT-OSS-20B"

    def generate(self, prompt, schema=None):

        if schema is not None:
            structured_model = self.model.with_structured_output(
                schema,
                method="json_mode"
            )

            response = structured_model.invoke(prompt)

            if isinstance(response, schema):
                return response

            return schema.model_validate(response)

        response = self.model.invoke(prompt)
        return response.content


    async def a_generate(self, prompt, schema=None):

        if schema is not None:
            structured_model = self.model.with_structured_output(
                schema,
                method="json_mode"
            )

            response = await structured_model.ainvoke(prompt)

            if isinstance(response, schema):
                return response

            return schema.model_validate(response)

        response = await self.model.ainvoke(prompt)
        return response.content


judge = GroqJudge(judge_model)


# 4. Initialize evaluation metrics
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
    async_mode=False,
    penalize_ambiguous_claims=True,
    evaluation_template=CustomFaithfulnessTemplate
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
    async_mode=False,
    evaluation_template=CustomAnswerRelevancyTemplate
)





def generator_eval(test_files=None):

    # 1. Load generated outputs
    with open(GENERATED_FILE, "r", encoding="utf-8") as file:
        generated_data = json.load(file)

    # 2. Filter records to match retriever evaluation
    if test_files is not None:

        test_files = set(test_files)

        generated_data = [
            record
            for record in generated_data
            if record["test_file"] in test_files
        ]

        found_files = {
            record["test_file"]
            for record in generated_data
        }

        missing_files = test_files - found_files

        if missing_files:
            print("\nMissing generated outputs:", missing_files)

    # 3. Evaluate each generated output
    evaluation_results = []

    for index, record in enumerate(generated_data):

        print(f"\n{'='*50}")
        print(f"Evaluating {index + 1}/{len(generated_data)}")
        print("Resume:", record["test_file"])
        print(f"{'='*50}")

        try:

            # Create DeepEval test case
            deepeval_case = LLMTestCase(
                input=record["input"],
                actual_output=record["actual_output"],
                retrieval_context=record["retrieval_context"]
            )

            print("\n========== EVALUATION INPUT ==========")
            print(record["input"])
            print("======================================\n")

            # =================================
            # FAITHFULNESS EVALUATION
            # =================================

            faithfulness_metric.measure(deepeval_case)

            faithfulness_score = faithfulness_metric.score
            faithfulness_reason = faithfulness_metric.reason

            print("\n--- Faithfulness ---")
            print("Score:", faithfulness_score)
            print("Reason:", faithfulness_reason)
            print("Passed:", faithfulness_metric.is_successful())

            for claim, verdict in zip(
                faithfulness_metric.claims,
                faithfulness_metric.verdicts
            ):
                print(f"\nClaim: {claim}")
                print(f"Verdict: {verdict.verdict}")

                if verdict.reason:
                    print(f"Reason: {verdict.reason}")

            time.sleep(10)

            # =================================
            # ANSWER RELEVANCY EVALUATION
            # =================================

            answer_relevancy_metric.measure(deepeval_case)

            answer_relevancy_score = answer_relevancy_metric.score
            answer_relevancy_reason = answer_relevancy_metric.reason

            print("\n--- Answer Relevancy ---")
            print("Score:", answer_relevancy_score)
            print("Reason:", answer_relevancy_reason)
            print("Passed:", answer_relevancy_metric.is_successful())

            # =================================
            # SCORE PREDICTION ERROR
            # =================================

            generated_output = json.loads(record["actual_output"])

            generated_score = generated_output["final_score"]
            human_score = record["human_score"]

            absolute_error = abs(generated_score - human_score)

            # =================================
            # STORE RESULTS
            # =================================

            evaluation_results.append({
                "test_file": record["test_file"],
                "human_score": human_score,
                "generated_score": generated_score,
                "absolute_error": absolute_error,
                "faithfulness": faithfulness_score,
                "faithfulness_reason": faithfulness_reason,
                "answer_relevancy": answer_relevancy_score,
                "answer_relevancy_reason": answer_relevancy_reason
            })

        except Exception as e:
            print(f"Error evaluating {record['test_file']}: {e}")

        # Wait before next resume
        if index < len(generated_data) - 1:
            print("\nWaiting 60 seconds...")
            time.sleep(60)

    # ==========================================
    # FINAL RESULTS
    # ==========================================

    results_df = pd.DataFrame(evaluation_results)

    print("\n========== GENERATOR EVALUATION RESULTS ==========")
    print(results_df.to_string(index=False))

    summary = {}

    if not results_df.empty:

        mae = results_df["absolute_error"].mean()
        faithfulness = results_df["faithfulness"].mean()
        answer_relevancy = results_df["answer_relevancy"].mean()

        summary = {
            "mae": mae,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy
        }

        print("\nGenerator MAE:", round(mae, 4))
        print("Average Faithfulness:", round(faithfulness, 4))
        print("Average Answer Relevancy:", round(answer_relevancy, 4))

    # Save evaluation results
    RESULTS_FILE = EVAL_DIR / "evaluation_results.csv"

    results_df.to_csv(RESULTS_FILE, index=False)

    print("\nEvaluation results saved to:", RESULTS_FILE)

    return {
        "individual_results": evaluation_results,
        "averages": summary
    }


# Allow standalone execution too
if __name__ == "__main__":
    generator_eval()
