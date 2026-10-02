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
from src.intent_router import classify_intent
from src.query_normalizer import normalize_query
from src.fallback_engine import handle_greeting, handle_out_of_scope, handle_vague


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Original factual benchmark (neural generation)
# ──────────────────────────────────────────────────────────────────────────────

NEURAL_BENCHMARK_TESTS = [
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


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Smart layer benchmark (intent router + normalizer + fallback)
# ──────────────────────────────────────────────────────────────────────────────

# --- 2a. Shorthand / Partial Query Tests (normalizer must expand & neural must answer)
SHORTHAND_TESTS = [
    {
        "category": "Shorthand – GPA",
        "input":    "gpa for admission",
        "expected_intent": "IN_DOMAIN_ADMISSION",
        "expected_normalized_contains": "gpa",
        "expected_keywords": ["3.0", "4.0"],
    },
    {
        "category": "Shorthand – Tuition",
        "input":    "tuition in state",
        "expected_intent": "IN_DOMAIN_ADMISSION",
        "expected_normalized_contains": "tuition",
        "expected_keywords": ["$18,500"],
    },
    {
        "category": "Shorthand – Deadline",
        "input":    "deadline for regular",
        "expected_intent": "IN_DOMAIN_ADMISSION",
        "expected_normalized_contains": "deadline",
        "expected_keywords": ["january 15th"],
    },
    {
        "category": "Shorthand – Scholarship",
        "input":    "any scholarships",
        "expected_intent": "IN_DOMAIN_ADMISSION",
        "expected_normalized_contains": "scholarship",
        "expected_keywords": ["merit", "scholarships"],
    },
    {
        "category": "Shorthand – Housing",
        "input":    "freshman housing",
        "expected_intent": "IN_DOMAIN_ADMISSION",
        "expected_normalized_contains": "housing",
        "expected_keywords": ["guaranteed", "first-year"],
    },
]

# --- 2b. Out-of-Scope / OOD Tests (must be caught by intent router)
OOD_TESTS = [
    {
        "category": "OOD – System Command",
        "input":    "clear ur data",
        "expected_intent": "OUT_OF_SCOPE",
    },
    {
        "category": "OOD – General Knowledge",
        "input":    "who is the president of the united states",
        "expected_intent": "OUT_OF_SCOPE",
    },
    {
        "category": "OOD – Cooking",
        "input":    "how do i bake chocolate cake recipe",
        "expected_intent": "OUT_OF_SCOPE",
    },
    {
        "category": "OOD – Programming",
        "input":    "write a python function to sort a list",
        "expected_intent": "OUT_OF_SCOPE",
    },
    {
        "category": "OOD – Sports",
        "input":    "who won the cricket world cup match today",
        "expected_intent": "OUT_OF_SCOPE",
    },
]

# --- 2c. Greeting / Chit-chat Tests (must be caught by intent router)
CHITCHAT_TESTS = [
    {
        "category": "Chit-chat – Greeting",
        "input":    "hello how are you doing today",
        "expected_intent": "GREETING_OR_CHITCHAT",
    },
    {
        "category": "Chit-chat – Identity",
        "input":    "who made you and what are you",
        "expected_intent": "GREETING_OR_CHITCHAT",
    },
    {
        "category": "Chit-chat – Farewell",
        "input":    "goodbye thank you see you later",
        "expected_intent": "GREETING_OR_CHITCHAT",
    },
    {
        "category": "Chit-chat – Compliment",
        "input":    "you are very helpful thank you",
        "expected_intent": "GREETING_OR_CHITCHAT",
    },
    {
        "category": "Chit-chat – Vague Single Word",
        "input":    "gpa",
        "expected_intent": "VAGUE_AMBIGUOUS",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def _section_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_neural_benchmark(model, tokenizer, context_length):
    _section_header("SECTION 1 — FACTUAL ACCURACY (Neural Generation, 12 Tests)")
    passed = 0
    total  = len(NEURAL_BENCHMARK_TESTS)

    for idx, test in enumerate(NEURAL_BENCHMARK_TESTS, 1):
        q        = test["question"]
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
        matched  = [k for k in expected if k.lower() in answer_lower]
        is_pass  = len(matched) > 0
        if is_pass:
            passed += 1

        status = "[PASS]" if is_pass else "[FAIL]"
        print(f"\n[{idx}/{total}] [{category}] {status}")
        print(f"  Prompt:   '{q}'")
        print(f"  Response: '{answer}'")
        print(f"  Matched:  {matched} (Expected any of: {expected})")

    accuracy = (passed / total) * 100
    print(f"\n  RESULT: {passed}/{total} passed ({accuracy:.1f}%)")
    return passed, total


def run_shorthand_benchmark(model, tokenizer, context_length):
    _section_header("SECTION 2a — SHORTHAND / PARTIAL QUERY HANDLING (5 Tests)")
    passed = 0
    total  = len(SHORTHAND_TESTS)

    for idx, test in enumerate(SHORTHAND_TESTS, 1):
        raw_input = test["input"]
        category  = test["category"]
        expected_intent = test["expected_intent"]
        expected_norm   = test["expected_normalized_contains"]
        expected_kw     = test["expected_keywords"]

        # Intent classification
        intent_result = classify_intent(raw_input)
        intent_ok = (intent_result.intent == expected_intent)

        # Query normalization
        normalized, was_expanded = normalize_query(raw_input)
        norm_ok = expected_norm.lower() in normalized.lower()

        # Neural generation on normalized query
        answer = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=normalized,
            max_new_tokens=60,
            context_length=context_length,
            temperature=0.25,
            top_k=5,
            repetition_penalty=1.1
        )

        answer_lower = answer.lower()
        matched  = [k for k in expected_kw if k.lower() in answer_lower]
        answer_ok = len(matched) > 0

        is_pass = intent_ok and norm_ok and answer_ok
        if is_pass:
            passed += 1

        status = "[PASS]" if is_pass else "[FAIL]"
        print(f"\n[{idx}/{total}] [{category}] {status}")
        print(f"  Raw Input:  '{raw_input}'")
        print(f"  Intent:     {intent_result.intent} (Expected: {expected_intent}) {'OK' if intent_ok else 'MISMATCH'}")
        print(f"  Normalized: '{normalized}' {'OK' if norm_ok else 'MISMATCH'}")
        print(f"  Response:   '{answer}'")
        print(f"  Matched:    {matched} (Expected any of: {expected_kw}) {'OK' if answer_ok else 'MISMATCH'}")

    accuracy = (passed / total) * 100
    print(f"\n  RESULT: {passed}/{total} passed ({accuracy:.1f}%)")
    return passed, total


def run_ood_benchmark():
    _section_header("SECTION 2b — OUT-OF-SCOPE DETECTION (5 Tests)")
    passed = 0
    total  = len(OOD_TESTS)

    for idx, test in enumerate(OOD_TESTS, 1):
        raw_input       = test["input"]
        category        = test["category"]
        expected_intent = test["expected_intent"]

        intent_result = classify_intent(raw_input)
        is_pass = (intent_result.intent == expected_intent)
        if is_pass:
            passed += 1

        status   = "[PASS]" if is_pass else "[FAIL]"
        fallback = handle_out_of_scope(raw_input) if is_pass else "(would have passed to neural model — wrong!)"
        print(f"\n[{idx}/{total}] [{category}] {status}")
        print(f"  Input:    '{raw_input}'")
        print(f"  Detected: {intent_result.intent} (Expected: {expected_intent})")
        print(f"  Fallback: '{fallback[:100]}...'")

    accuracy = (passed / total) * 100
    print(f"\n  RESULT: {passed}/{total} passed ({accuracy:.1f}%)")
    return passed, total


def run_chitchat_benchmark():
    _section_header("SECTION 2c — GREETING & CHIT-CHAT DETECTION (5 Tests)")
    passed = 0
    total  = len(CHITCHAT_TESTS)

    for idx, test in enumerate(CHITCHAT_TESTS, 1):
        raw_input       = test["input"]
        category        = test["category"]
        expected_intent = test["expected_intent"]

        intent_result = classify_intent(raw_input)
        is_pass = (intent_result.intent == expected_intent)
        if is_pass:
            passed += 1

        status = "[PASS]" if is_pass else "[FAIL]"

        if expected_intent == "GREETING_OR_CHITCHAT":
            response = handle_greeting(raw_input)
        else:
            response = handle_vague(raw_input)

        print(f"\n[{idx}/{total}] [{category}] {status}")
        print(f"  Input:    '{raw_input}'")
        print(f"  Detected: {intent_result.intent} (Expected: {expected_intent})")
        print(f"  Response: '{response[:110]}...'")

    accuracy = (passed / total) * 100
    print(f"\n  RESULT: {passed}/{total} passed ({accuracy:.1f}%)")
    return passed, total


def run_benchmark(checkpoint_path="checkpoints/tinygpt_admissions.pt"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' does not exist.")
        print("Please run train_admissions.py first.")
        sys.exit(1)

    print("=" * 70)
    print("COLLEGE ADMISSIONS SMART BOT — FULL BENCHMARK SUITE")
    print("=" * 70)

    model, tokenizer = load_checkpoint(
        path=checkpoint_path,
        model_class=TinyGPT,
        tokenizer_class=Tokenizer
    )
    context_length = model.max_context_length

    # Run all sections
    p1, t1 = run_neural_benchmark(model, tokenizer, context_length)
    p2, t2 = run_shorthand_benchmark(model, tokenizer, context_length)
    p3, t3 = run_ood_benchmark()
    p4, t4 = run_chitchat_benchmark()

    # Grand total
    total_passed = p1 + p2 + p3 + p4
    total_tests  = t1 + t2 + t3 + t4
    accuracy     = (total_passed / total_tests) * 100

    print("\n" + "=" * 70)
    print("FULL BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Section 1 — Factual Accuracy:         {p1}/{t1} ({p1/t1*100:.1f}%)")
    print(f"  Section 2a — Shorthand Query Handling:{p2}/{t2} ({p2/t2*100:.1f}%)")
    print(f"  Section 2b — OOD Detection:           {p3}/{t3} ({p3/t3*100:.1f}%)")
    print(f"  Section 2c — Greeting/Chit-chat:      {p4}/{t4} ({p4/t4*100:.1f}%)")
    print("-" * 70)
    print(f"  OVERALL RESULT: {total_passed}/{total_tests} tests passed ({accuracy:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
