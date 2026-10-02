import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.model import TinyGPT
from src.generation import generate_text
from src.checkpoint import load_checkpoint
from src.tokenizer import Tokenizer


BENCHMARK_TESTS = [
    {
        "category": "Requirements",
        "question": "what is the minimum gpa for admission?",
        "expected_keywords": ["3.0", "4.0"]
    },
    {
        "category": "Testing Policy",
        "question": "is the sat or act required?",
        "expected_keywords": ["test-optional", "optional"]
    },
    {
        "category": "Deadlines",
        "question": "when is the regular decision deadline?",
        "expected_keywords": ["january 15th"]
    },
    {
        "category": "Deadlines",
        "question": "when is the early action deadline?",
        "expected_keywords": ["november 1st"]
    },
    {
        "category": "Tuition & Costs",
        "question": "how much is tuition?",
        "expected_keywords": ["$18,500", "$32,000"]
    },
    {
        "category": "Tuition & Costs",
        "question": "what is the application fee?",
        "expected_keywords": ["$65"]
    },
    {
        "category": "Financial Aid",
        "question": "do you offer scholarships?",
        "expected_keywords": ["merit", "scholarships", "$3,000"]
    },
    {
        "category": "Financial Aid",
        "question": "what is the school code for fafsa?",
        "expected_keywords": ["001234"]
    },
    {
        "category": "Campus Life",
        "question": "is on-campus housing guaranteed?",
        "expected_keywords": ["guaranteed", "first-year"]
    },
    {
        "category": "Programs",
        "question": "tell me about the computer science major.",
        "expected_keywords": ["computer science", "artificial intelligence"]
    },
    {
        "category": "Contact",
        "question": "how do i contact the admissions office?",
        "expected_keywords": ["admissions@university.edu", "1-800-555-0199"]
    },
    {
        "category": "International",
        "question": "what english test scores are accepted?",
        "expected_keywords": ["toefl", "ielts"]
    },
]


def run_benchmark(checkpoint_path="checkpoints/tinygpt_admissions.pt"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' does not exist.")
        print("Please run train_admissions.py first.")
        sys.exit(1)

    print("=" * 70)
    print("COLLEGE ADMISSIONS BOT - BENCHMARK & FACTUAL VERIFICATION")
    print("=" * 70)

    model, tokenizer = load_checkpoint(
        path=checkpoint_path,
        model_class=TinyGPT,
        tokenizer_class=Tokenizer
    )
    context_length = model.max_context_length

    passed = 0
    total = len(BENCHMARK_TESTS)

    for idx, test in enumerate(BENCHMARK_TESTS, 1):
        q = test["question"]
        expected = test["expected_keywords"]
        category = test["category"]

        answer = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            max_new_tokens=60,
            context_length=context_length,
            temperature=0.25,
            top_k=5,
            repetition_penalty=1.1
        )

        answer_lower = answer.lower()
        matched = [k for k in expected if k.lower() in answer_lower]
        is_pass = len(matched) > 0

        if is_pass:
            passed += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"

        print(f"\n[{idx}/{total}] [{category}] {status}")
        print(f"Prompt:   '{q}'")
        print(f"Response: '{answer}'")
        print(f"Matched:  {matched} (Expected any of: {expected})")

    accuracy = (passed / total) * 100
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULT: {passed}/{total} tests passed ({accuracy:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
