import sys
import os
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.model import TinyGPT
from src.generation import generate_text
from src.checkpoint import load_checkpoint
from src.tokenizer import Tokenizer
from main import show_debug_info


def print_banner():
    print("=" * 68)
    print("UNIVERSITY OFFICE OF UNDERGRADUATE ADMISSIONS")
    print("Virtual Admissions & Enrollment Assistant (TinyGPT)")
    print("=" * 68)
    print("Welcome! I am here to help answer questions about admission requirements,")
    print("application deadlines, tuition & aid, academic majors, and campus life.\n")
    print("Helpful Shortcuts:")
    print("  faq          -> View frequently asked questions")
    print("  deadlines    -> View application deadlines at a glance")
    print("  contact      -> View admissions office contact details & hours")
    print("  debug <text> -> Inspect tokenization and neural attention")
    print("  quit         -> Exit the assistant")
    print("=" * 68)


def print_faq():
    print("\n" + "-" * 55)
    print("[FAQ] Frequently Asked Admissions Questions:")
    print("-" * 55)
    print(" * What is the minimum GPA for admission?")
    print(" * When is the application deadline?")
    print(" * How much is tuition and total cost?")
    print(" * Do you offer scholarships and financial aid?")
    print(" * Is SAT or ACT required?")
    print(" * What majors do you offer?")
    print(" * Is on-campus housing guaranteed?")
    print(" * How do I contact the admissions office?")
    print("-" * 55 + "\n")


def print_deadlines():
    print("\n" + "-" * 55)
    print("[DEADLINES] Key Application Deadlines:")
    print("-" * 55)
    print(" * Early Action (Non-Binding):   November 1st")
    print(" * Early Decision (Binding):     November 15th")
    print(" * Regular Decision:             January 15th")
    print(" * Priority Financial Aid (FAFSA): February 1st")
    print(" * Decision Release:             Mid-December (Early) / Late March (Regular)")
    print(" * National Candidate Reply Day: May 1st ($300 deposit)")
    print("-" * 55 + "\n")


def print_contact():
    print("\n" + "-" * 55)
    print("[CONTACT] Admissions Office Contact Information:")
    print("-" * 55)
    print(" * Email:        admissions@university.edu")
    print(" * Toll-Free:    1-800-555-0199")
    print(" * Local Phone:  555-0123")
    print(" * Office Hours: Mon - Fri, 8:30 AM - 5:00 PM EST")
    print(" * Location:     Hall Hall, Room 101, 100 University Ave")
    print(" * Portal:       Check status online 24/7 at applicant portal")
    print("-" * 55 + "\n")


def main():
    checkpoint_path = "checkpoints/tinygpt_admissions.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'.")
        print("Please run 'python train_admissions.py' first to train the admissions model.")
        sys.exit(1)

    print(f"Loading admissions model from: {checkpoint_path} ...")
    model, tokenizer = load_checkpoint(
        path=checkpoint_path,
        model_class=TinyGPT,
        tokenizer_class=Tokenizer
    )
    context_length = model.max_context_length

    print_banner()

    while True:
        try:
            prompt = input("Applicant: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye and best of luck with your college journey!")
            break

        if not prompt:
            continue

        lower = prompt.lower()

        if lower in ["quit", "exit"]:
            print("\nThank you for consulting the Admissions Office. Goodbye!")
            break

        if lower == "faq":
            print_faq()
            continue

        if lower == "deadlines":
            print_deadlines()
            continue

        if lower == "contact":
            print_contact()
            continue

        if lower.startswith("debug "):
            debug_text = prompt[6:].strip()
            show_debug_info(
                model=model,
                tokenizer=tokenizer,
                text=debug_text,
                context_length=context_length
            )
            continue

        # Generate response using factual parameters
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=65,
            context_length=context_length,
            temperature=0.3,
            top_k=5,
            repetition_penalty=1.15
        )

        print(f"\nAdmissions Bot: {response}\n")


if __name__ == "__main__":
    main()
