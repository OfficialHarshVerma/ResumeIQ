
from pathlib import Path
import json
import sys
import math

# ==========================================
# PATHS
# ==========================================

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
GOLDEN_FILE = EVAL_DIR / "retriever_golden_dataset.json"

# Import existing retriever
sys.path.insert(0, str(PROJECT_DIR))

from user import retrieve


# ==========================================
# METRIC CALCULATION
# ==========================================


def calculate_metrics(retrieved_fingerprints, relevant_fingerprints, k):

    top_k = retrieved_fingerprints[:k]

    relevance = [
        1 if fingerprint in relevant_fingerprints else 0
        for fingerprint in top_k
    ]

    relevant_count = sum(relevance)

    # Capped-three golden coverage
    recall_denominator = min(len(relevant_fingerprints), 3)

    # Precision@k
    precision = relevant_count / k

    # Recall@k (capped at 3 relevant resumes)
    recall = (
        min(relevant_count, 3) / recall_denominator
        if recall_denominator
        else 0.0
    )

    # MRR@k
    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            reciprocal_rank = 1 / rank
            break

    # NDCG@k
    dcg = sum(
        is_relevant / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevance, start=1)
    )

    ideal_relevant_count = min(len(relevant_fingerprints), k)

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg
    }




# ==========================================
# RETRIEVER EVALUATION
# ==========================================

def retrieve_eval():

    with open(GOLDEN_FILE, "r", encoding="utf-8") as file:
        golden_data = json.load(file)

    results = []

    k_values = [5]

    # Evaluate all test resumes
    for query in golden_data["queries"]:

        test_resume = EVAL_DIR / "test_resumes" / query["test_file"]

        relevant_fingerprints = set(
            query["relevant_benchmark_fingerprints"]
        )

        # Retrieve benchmark resumes
        docs, candidate_resume_json = retrieve(test_resume)

        retrieved_fingerprints = [
            doc["fingerprint"]
            for doc in docs
        ]

        query_results = {
            "test_file": query["test_file"],
            "retrieved_count": len(retrieved_fingerprints),
            "metrics": {}
        }

        print(f"\n========== {query['test_file']} ==========")
        print(f"Retrieved resumes: {len(retrieved_fingerprints)}")

        # Calculate metrics at @3 and @5
        for k in k_values:

            metrics = calculate_metrics(
                retrieved_fingerprints,
                relevant_fingerprints,
                k
            )

            query_results["metrics"][f"@{k}"] = metrics

            print(f"\n--- Metrics @{k} ---")
            print(f"Precision@{k}: {metrics['precision']:.4f}")
            print(f"Recall@{k}:    {metrics['recall']:.4f}")
            print(f"MRR@{k}:       {metrics['mrr']:.4f}")
            print(f"NDCG@{k}:      {metrics['ndcg']:.4f}")

        results.append(query_results)

    # ==========================================
    # OVERALL AVERAGES
    # ==========================================

    print("\n========== OVERALL RETRIEVER RESULTS ==========")
    print(f"Resumes evaluated: {len(results)}")

    averages = {}

    for k in k_values:

        averages[f"@{k}"] = {}

        print(f"\n--- Average Metrics @{k} ---")

        for metric in ["precision", "recall", "mrr", "ndcg"]:

            average = sum(
                result["metrics"][f"@{k}"][metric]
                for result in results
            ) / len(results) if results else 0.0

            averages[f"@{k}"][metric] = average

            print(f"Average {metric.upper()}@{k}: {average:.4f}")

    return {
        "individual_results": results,
        "averages": averages
    }
