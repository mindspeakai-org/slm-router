import sys
from pathlib import Path

# Ensure src is in sys.path
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from slm_router.router import Router

TEST_CASES = [
    # 1. LOCAL - General Knowledge & Math (no memory)
    ("What is 2 + 2?", "LOCAL", False, None),
    ("Tell me a joke.", "LOCAL", False, None),
    ("Explain photosynthesis in simple terms.", "LOCAL", False, None),

    # 2. LOCAL - Device Commands (no memory)
    ("Turn on the light.", "LOCAL", False, None),
    ("Turn off the fan.", "LOCAL", False, None),
    ("Open the door.", "LOCAL", False, None),

    # 3. LOCAL - Single Memory Request
    ("What is my favorite animal?", "LOCAL", True, [["favorite_animal", "animal"]]),
    ("What is my name?", "LOCAL", True, [["child_name", "user_name", "name"]]),

    # 4. LOCAL - Multiple Memory Requests in Single Call
    ("What is my name and what is my favorite animal?", "LOCAL", True, [["name", "child_name", "user_name"], ["favorite_animal", "animal"]]),

    # 5. CLOUD - Live/Real-time information (no memory)
    ("What is the weather today?", "CLOUD", False, None),

    # 6. CLOUD - Long-form / Heavy tasks (no memory)
    ("Tell me a 500-word story about a dragon.", "CLOUD", False, None),
    ("Write a 3000-word essay about climate change.", "CLOUD", False, None),
    ("Develop a detailed production-ready distributed system architecture.", "CLOUD", False, None),

    # 7. CLOUD - Long-form with Memory
    ("Write a 500-word story about my favorite animal.", "CLOUD", True, [["favorite_animal", "animal"]]),
]


def main():
    router = Router()

    total_tests = len(TEST_CASES)
    correct = 0
    incorrect = 0

    print("==================================================")
    print("STARTING SLM-ROUTER DECISION BENCHMARK TEST")
    print("==================================================")

    for i, (query, exp_proc, exp_mem, exp_keys) in enumerate(TEST_CASES, start=1):
        decision = router.route(query)

        proc = decision.get("processing")
        mem_req = decision.get("memory_required")
        mem_req_obj = decision.get("memory_request")

        passed_proc = (proc == exp_proc)
        passed_mem = (mem_req == exp_mem)

        passed_keys = True
        if exp_mem:
            if not mem_req_obj or not mem_req_obj.get("keys"):
                passed_keys = False
            else:
                actual_keys = [k.lower() for k in mem_req_obj.get("keys", [])]
                # Each expected concept must have at least one matching alias in actual_keys
                for concept_aliases in exp_keys:
                    if isinstance(concept_aliases, str):
                        concept_aliases = [concept_aliases]
                    if not any(
                        any(alias.lower() in ak or ak in alias.lower() for alias in concept_aliases)
                        for ak in actual_keys
                    ):
                        passed_keys = False
                        break
        else:
            if mem_req_obj is not None:
                passed_keys = False

        passed = passed_proc and passed_mem and passed_keys

        if passed:
            correct += 1
        else:
            incorrect += 1

        status = "PASS" if passed else "FAIL"

        print(f"\nTEST {i}")
        print(f"Query: {query}")
        print(f"Decision: {decision}")
        print(f"Expected: processing={exp_proc}, memory_required={exp_mem}, keys={exp_keys}")
        print(f"Status: {status}")

    accuracy = (correct / total_tests) * 100.0

    print("\n================ SUMMARY ================")
    print(f"Total Tests: {total_tests}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("==========================================")

    if incorrect > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
