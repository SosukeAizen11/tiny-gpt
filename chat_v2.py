import torch

from src.model import TinyGPT
from src.generation import generate_text
from src.checkpoint import load_checkpoint
from src.tokenizer import Tokenizer


def main():

    checkpoint_path = (
        "checkpoints/tinygpt_v2.pt"
    )

    # --------------------------------
    # Load trained model
    # --------------------------------

    model, tokenizer = load_checkpoint(
        path=checkpoint_path,
        model_class=TinyGPT,
        tokenizer_class=Tokenizer
    )

    context_length = (
        model.max_context_length
    )

    print("=" * 60)
    print("TinyGPT V2 - Chat")
    print("=" * 60)

    print(
        "\nLoaded trained model."
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"Context length: "
        f"{context_length}"
    )

    print(
        "\nType 'quit' to exit."
    )

    # --------------------------------
    # Interactive generation
    # --------------------------------

    while True:

        prompt = input("\nYou: ").strip()

        if prompt.lower() == "quit":
            print("Goodbye!")
            break

        if not prompt:
            continue

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=30,
            context_length=context_length,
            temperature=0.4,
            top_k=20
        )

        print(
            f"TinyGPT: {generated}"
        )


if __name__ == "__main__":
    main()
