import json
from pathlib import Path
import sys


# ---------------- PATHS ----------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

TEST_DIR = PROJECT_DIR / "evals" / "test_resumes"

OUTPUT_FILE = PROJECT_DIR / "evals" / "pipeline_outputs.json"

ROLE_NAME = "AI Engineer"


from user import generate

# ---------------- TEST DATA ----------------

# Human-assigned scores for the 8 evaluation resumes.
HUMAN_SCORES = {
    "weak_08.pdf": 2.2,
    "weak_09.pdf": 1.5,
    "weak_10.pdf": 1.8,
    "average_08.pdf": 5.5,
    "average_09.pdf": 5.2,
    "average_10.pdf": 6.0,
    "strong_08.pdf": 8.3,
    "strong_09.pdf": 8.1,
    "strong_10.pdf": 8.8
}


# ---------------- PIPELINE EXPORT ----------------

def export_pipeline_outputs():

    exported_data = []

    for index, (filename, human_score) in enumerate(
        HUMAN_SCORES.items(), start=1
    ):

        resume_path = TEST_DIR / filename

        print("\n" + "=" * 60)
        print(f"Processing {index}/{len(HUMAN_SCORES)}")
        print("Resume:", filename)
        print("=" * 60)

        if not resume_path.exists():
            print("Resume not found. Skipping:", resume_path)
            continue

        try:

            # Run the actual pipeline
            result, retrieval_context = generate(
                Path=resume_path,
                role_name=ROLE_NAME,
                return_context=True
            )

            # Convert Pydantic output into JSON-compatible dictionary
            if hasattr(result, "model_dump"):
                actual_output = json.dumps(
                    result.model_dump(),
                    ensure_ascii=False
                )
            else:
                actual_output = json.dumps(
                    result,
                    ensure_ascii=False
                )

            # Input given to the generator
            input_text = (
                f"Evaluate the candidate's suitability for the "
                f"{ROLE_NAME} role using the candidate resume, "
                f"role guide, and retrieved benchmark resume. "
                f"Apply the specified scoring rubric."
            )

            record = {
                "test_file": filename,
                "role_name": ROLE_NAME,
                "input": input_text,
                "actual_output": actual_output,
                "retrieval_context": retrieval_context,
                "human_score": human_score
            }

            exported_data.append(record)

            print("Successfully exported:", filename)

        except Exception as e:

            print(f"Error processing {filename}: {e}")

    # ---------------- SAVE RESULTS ----------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            exported_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("PIPELINE EXPORT COMPLETED")
    print("=" * 60)

    print("Successfully exported:", len(exported_data))
    print("Expected resumes:", len(HUMAN_SCORES))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    export_pipeline_outputs()