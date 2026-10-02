import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.tokenizer import Tokenizer
from src.dataset import LanguageModelDataset
from src.model import TinyGPT
from src.generation import generate_text
from src.checkpoint import save_checkpoint


def train_admissions_bot(
    data_path="data/admissions_data.txt",
    checkpoint_path="checkpoints/tinygpt_admissions.pt",
    context_length=128,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
    ff_hidden_dim=512,
    batch_size=16,
    max_epochs=350,
    target_loss=0.15
):
    print("=" * 65)
    print("TinyGPT - College Admissions Bot Training Pipeline")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Admissions dataset not found at {data_path}")

    # 1. Load Dataset
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    print(f"\n[+] Loaded {len(blocks)} conversation dialogues from {data_path}")

    # 2. Tokenizer Setup
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(blocks)
    print(f"[+] Vocabulary size: {tokenizer.vocab_size} tokens")

    # 3. Language Model Dataset
    dataset = LanguageModelDataset(
        texts=blocks,
        tokenizer=tokenizer,
        context_length=context_length
    )
    print(f"[+] Total training sequences created: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("No training sequences generated from dataset.")

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # 4. Initialize Scaled Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Training device: {device}")

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_hidden_dim=ff_hidden_dim,
        max_context_length=context_length
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model Architecture: {num_layers} layers, {num_heads} heads, {embedding_dim} dim, {ff_hidden_dim} FFN")
    print(f"[+] Model Parameters: {total_params:,} total ({trainable_params:,} trainable)")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-5
    )

    pad_id = tokenizer.token_to_id["<PAD>"]

    # 6. Training Loop
    print("\n" + "=" * 65)
    print("Starting Training...")
    print("=" * 65)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits, _ = model(input_ids)

            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                target_ids.view(-1),
                ignore_index=pad_id
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if epoch % 10 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{max_epochs} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        if avg_loss < target_loss:
            print(f"\n[OK] Target loss reached at Epoch {epoch} (Loss: {avg_loss:.4f} < {target_loss})")
            break

    # 7. Save Checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        path=checkpoint_path
    )
    print(f"\n[OK] Admissions Bot checkpoint successfully saved to: {checkpoint_path}")

    # 8. Sample Inference Smoke Test
    print("\n" + "=" * 65)
    print("Running Sample Verification Inquiries...")
    print("=" * 65)

    test_questions = [
        "what is the minimum gpa for admission?",
        "when is the application deadline?",
        "how much is tuition?",
        "do you offer scholarships?",
        "is on-campus housing guaranteed?",
        "how do i contact the admissions office?"
    ]

    for q in test_questions:
        ans = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            max_new_tokens=50,
            context_length=context_length,
            temperature=0.3,
            top_k=5
        )
        print(f"\nQ: {q}")
        print(f"A: {ans}")

    print("\n" + "=" * 65)
    print("[OK] Training and verification complete!")
    print("=" * 65)


if __name__ == "__main__":
    train_admissions_bot()
