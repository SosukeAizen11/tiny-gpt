import os

import torch


def save_checkpoint(
    model,
    tokenizer,
    path
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token,
        "vocab_size": tokenizer.vocab_size,
        "model_config": {
            "embedding_dim": model.embedding.embedding.embedding_dim,
            "num_heads": model.transformer_blocks[
                0
            ].attention.num_heads,
            "num_layers": len(
                model.transformer_blocks
            ),
            "ff_hidden_dim": model.transformer_blocks[
                0
            ].feed_forward.network[0].out_features,
            "max_context_length":
                model.max_context_length,
        }
    }

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    model_class,
    tokenizer_class
):
    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    tokenizer = tokenizer_class()

    tokenizer.token_to_id = (
        checkpoint["token_to_id"]
    )

    tokenizer.id_to_token = {
        int(key): value
        for key, value
        in checkpoint["id_to_token"].items()
    }

    config = checkpoint["model_config"]

    model = model_class(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_hidden_dim=config["ff_hidden_dim"],
        max_context_length=config[
            "max_context_length"
        ],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, tokenizer