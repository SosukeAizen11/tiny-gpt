import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.tokenizer import Tokenizer
from src.dataset import LanguageModelDataset
from src.model import TinyGPT
from src.generation import generate_text
from src.checkpoint import save_checkpoint


def show_debug_info(
    model,
    tokenizer,
    text,
    context_length
):
    print("\n" + "=" * 60)
    print("DEBUG MODE")
    print("=" * 60)

    # ==============================================
    # TOKENIZATION
    # ==============================================

    tokens = tokenizer.tokenize(text)

    print("\nINPUT")
    print(text)

    print("\nTOKENS")
    print(tokens)

    # ==============================================
    # TOKEN IDS
    # ==============================================

    token_ids = tokenizer.encode(text)

    print("\nTOKEN IDS")
    print(token_ids)

    if not token_ids:
        print("No valid tokens.")
        return

    # Keep only the latest context tokens
    token_ids = token_ids[-context_length:]

    input_tensor = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    # ==============================================
    # FORWARD PASS
    # ==============================================

    model.eval()

    with torch.no_grad():

        (
            logits,
            attention_weights,
            debug_info
        ) = model(
            input_tensor,
            return_debug=True
        )

    # ==============================================
    # MODEL PIPELINE
    # ==============================================

    print("\nMODEL PIPELINE")

    print(
        "Token Embeddings:      ",
        list(
            debug_info[
                "token_embeddings"
            ].shape
        )
    )

    print(
        "Positioned Embeddings: ",
        list(
            debug_info[
                "positioned_embeddings"
            ].shape
        )
    )

    for index in range(
        len(model.transformer_blocks)
    ):

        print(
            f"Transformer Block {index + 1}:   ",
            list(
                debug_info[
                    f"transformer_block_{index + 1}"
                ].shape
            )
        )

    print(
        "Final LayerNorm:       ",
        list(
            debug_info[
                "final_layer_norm"
            ].shape
        )
    )

    print(
        "Logits:                ",
        list(
            debug_info[
                "logits"
            ].shape
        )
    )

    # ==============================================
    # ATTENTION
    # ==============================================

    print("\nATTENTION")

    for layer_index, weights in enumerate(
        attention_weights
    ):

        print(
            f"\nLayer {layer_index + 1}"
        )

        print(
            f"Shape: {list(weights.shape)}"
        )

        for head_index in range(
            weights.shape[0]
        ):

            print(
                f"  Head {head_index + 1}: "
                f"{list(weights[head_index].shape)}"
            )

    # ==============================================
    # NEXT TOKEN PREDICTION
    # ==============================================

    last_logits = logits[-1]

    probabilities = F.softmax(
        last_logits,
        dim=-1
    )

    print("\nNEXT TOKEN PREDICTION")

    print(
        "Logits shape:",
        list(last_logits.shape)
    )

    print(
        "Probability shape:",
        list(probabilities.shape)
    )

    # ==============================================
    # TOP PREDICTIONS
    # ==============================================

    top_k = min(
        5,
        tokenizer.vocab_size
    )

    top_probs, top_ids = torch.topk(
        probabilities,
        k=top_k
    )

    print("\nTOP PREDICTIONS")

    for probability, token_id in zip(
        top_probs,
        top_ids
    ):

        token = tokenizer.id_to_token[
            token_id.item()
        ]

        percentage = (
            probability.item() * 100
        )

        print(
            f"{token:<15}"
            f"{percentage:>8.2f}%"
        )

    # ==============================================
    # SELECTED TOKEN
    # ==============================================

    selected_id = torch.argmax(
        probabilities
    ).item()

    selected_token = tokenizer.id_to_token[
        selected_id
    ]

    print("\nSELECTED TOKEN")

    print(
        f"{selected_token} "
        f"(ID: {selected_id})"
    )


def main():

    # ==============================================
    # LOAD DATASET
    # ==============================================

    with open(
        "data/conversations.txt",
        "r",
        encoding="utf-8"
    ) as file:

        conversation_text = file.read()

    texts = [
        block.strip()
        for block in conversation_text.split("\n\n")
        if block.strip()
    ]

    # ==============================================
    # TOKENIZER
    # ==============================================

    tokenizer = Tokenizer()

    tokenizer.build_vocabulary(
        texts
    )

    print("=" * 60)
    print("TinyGPT V2")
    print("=" * 60)

    print(
        "\nVocabulary size:",
        tokenizer.vocab_size
    )

    # ==============================================
    # CONTEXT LENGTH
    # ==============================================

    context_length = 64

    print(
        "Context length:",
        context_length
    )

    # ==============================================
    # DATASET
    # ==============================================

    dataset = LanguageModelDataset(
        texts=texts,
        tokenizer=tokenizer,
        context_length=context_length
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset contains zero training examples."
        )

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    print(
        "Training examples:",
        len(dataset)
    )

    # ==============================================
    # MODEL
    # ==============================================

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        ff_hidden_dim=256,
        max_context_length=context_length
    )

    # ==============================================
    # PARAMETERS
    # ==============================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nMODEL PARAMETERS")

    print(
        f"Total:     {total_parameters:,}"
    )

    print(
        f"Trainable: {trainable_parameters:,}"
    )

    # ==============================================
    # OPTIMIZER
    # ==============================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )

    # ==============================================
    # TRAINING
    # ==============================================

    epochs = 300

    # Cosine annealing scheduler for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    print(
        f"\nEpochs: {epochs}"
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for input_ids, target_ids in train_loader:

            logits, _ = model(input_ids)

            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                target_ids.view(-1),
                ignore_index=tokenizer.token_to_id["<PAD>"]
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        average_loss = total_loss / len(train_loader)

        # Print every 10 epochs to reduce noise
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d} | "
                f"Loss: {average_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # Early stopping: model has converged well enough
        if average_loss < 0.3:
            print(
                f"\nEarly stop at epoch {epoch + 1} "
                f"(loss={average_loss:.4f} < 0.3)"
            )
            break

    print("\nTraining complete!")

    # ==============================================
    # SAVE CHECKPOINT
    # ==============================================

    checkpoint_path = (
        "checkpoints/tinygpt_v2.pt"
    )

    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        path=checkpoint_path
    )

    print(
        f"\nModel saved to:"
        f"\n{checkpoint_path}"
    )

    # ==============================================
    # INTERACTIVE MODE
    # ==============================================

    print("\n" + "=" * 60)
    print("TinyGPT Interactive Mode")
    print("=" * 60)

    print(
        "\nCommands:"
    )

    print(
        "  debug <text>  -> inspect model internals"
    )

    print(
        "  quit          -> exit"
    )

    print(
        "\nYou can now talk to TinyGPT."
    )

    # ==============================================
    # CHAT LOOP
    # ==============================================

    while True:

        prompt = input("\nYou: ").strip()

        # ------------------------------------------
        # Quit
        # ------------------------------------------

        if prompt.lower() == "quit":

            print(
                "Goodbye!"
            )

            break

        # ------------------------------------------
        # Empty input
        # ------------------------------------------

        if not prompt:
            continue

        # ------------------------------------------
        # Debug mode
        # ------------------------------------------

        if prompt.lower().startswith(
            "debug "
        ):

            debug_text = prompt[
                6:
            ].strip()

            show_debug_info(
                model=model,
                tokenizer=tokenizer,
                text=debug_text,
                context_length=context_length
            )

            continue

        # ------------------------------------------
        # Normal generation
        # ------------------------------------------

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