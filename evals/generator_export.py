
import json
from pathlib import Path
import sys
import time

# 1. Define paths
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent

GOLDEN_FILE = EVAL_DIR / "generator_golden_dataset.json"
TEST_RESUME_DIR = EVAL_DIR / "test_resumes"
OUTPUT_FILE = EVAL_DIR / "generated_outputs.json"

# 2. Import the existing resume generator
sys.path.insert(0, str(PROJECT_DIR))

from user import generate


# 3. Load the golden dataset
with open(GOLDEN_FILE, "r", encoding="utf-8") as file:
    golden_data = json.load(file)


# 4. Store all generated outputs here
generated_outputs = []

# 5. Process each test resume
for index, test_case in enumerate(golden_data):

    print(f"\n{'='*50}")
    print(f"Generating {index + 1}/{len(golden_data)}: {test_case['test_file']}")
    print(f"{'='*50}")

    try:
        # Get resume path
        resume_path = TEST_RESUME_DIR / test_case["test_file"]

        # Run resume generator and capture retrieval context
        result, retrieval_context = generate(
            Path=resume_path,
            role_name=test_case["role_name"],
            return_context=True
        )

        # Get generated output
        actual_output = result.model_dump_json(indent=2)

        # Store the generated record
        record = {
            "test_file": test_case["test_file"],
            "role_name": test_case["role_name"],
            "input": test_case["input"],
            "actual_output": actual_output,
            "retrieval_context": retrieval_context,

            # Used only for calculating MAE later.
            # Never passed to the generator or judge.
            "human_score": test_case["human_score"]
        }

        generated_outputs.append(record)

        print("\nGenerated Score:", result.final_score)
        print("Successfully processed:", test_case["test_file"])

    except Exception as e:
        print(f"Error processing {test_case['test_file']}: {e}")

    # Wait between Groq requests
    if index < len(golden_data) - 1:
        print("\nWaiting 60 seconds before next resume...")
        time.sleep(60)


# 6. Save all generated records into ONE JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        generated_outputs,
        file,
        indent=4,
        ensure_ascii=False
    )


# 7. Final summary
print("\n========== GENERATOR EXPORT COMPLETED ==========")
print("Total resumes:", len(golden_data))
print("Successfully generated:", len(generated_outputs))
print("Failed:", len(golden_data) - len(generated_outputs))
print("Output file:", OUTPUT_FILE)
