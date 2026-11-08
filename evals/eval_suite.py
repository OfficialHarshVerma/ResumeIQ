
from retriever_eval import retrieve_eval
from generator_evaluator import generator_eval
from pipeline_evaluator import pipeline_eval
from application_evaluator import evaluate_application


def main():

    print("\n========== EVALUATION SUITE ==========")

    # =================================
    # 1. RETRIEVER EVALUATION
    # =================================

    print("\n\nRUNNING RETRIEVER EVALUATION")

    retriever_results = retrieve_eval()

    processed_files = [
        result["test_file"]
        for result in retriever_results["individual_results"]
    ]

    # =================================
    # 2. GENERATOR EVALUATION
    # =================================

    print("\n\nRUNNING GENERATOR EVALUATION")

    generator_results = generator_eval(
        test_files=processed_files
    )

    # =================================
    # 3. PIPELINE EVALUATION
    # =================================

    print("\n\nRUNNING PIPELINE EVALUATION")

    pipeline_results = pipeline_eval(
        test_files=processed_files
    )

    # =================================
    # 4. APPLICATION EVALUATION
    # =================================

    print("\n\nRUNNING APPLICATION EVALUATION")

    application_results = evaluate_application(
        test_files=processed_files
    )

    print("\n========== EVALUATION SUITE COMPLETED ==========")


if __name__ == "__main__":
    main()
