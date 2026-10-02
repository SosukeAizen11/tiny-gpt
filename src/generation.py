import torch
import torch.nn.functional as F


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=60,
    context_length=128,
    temperature=0.3,
    top_k=5,
    repetition_penalty=1.1
):
    model.eval()

    bos_id = tokenizer.token_to_id["<BOS>"]
    user_id = tokenizer.token_to_id["<USER>"]
    assistant_id = tokenizer.token_to_id["<ASSISTANT>"]
    eos_id = tokenizer.token_to_id["<EOS>"]
    pad_id = tokenizer.token_to_id["<PAD>"]

    # IMPORTANT:
    # Match the exact structure used during training.
    formatted_prompt = (
        "<BOS> "
        "<USER> "
        + prompt
        + " "
        "<ASSISTANT>"
    )

    generated_ids = tokenizer.encode(
        formatted_prompt
    )

    prompt_length = len(generated_ids)

    for _ in range(max_new_tokens):

        context_ids = generated_ids[
            -context_length:
        ]

        input_tensor = torch.tensor(
            context_ids,
            dtype=torch.long
        )

        with torch.no_grad():
            logits, _ = model(input_tensor)

        next_token_logits = logits[-1].clone()

        # Repetition penalty on newly generated response tokens
        if repetition_penalty is not None and repetition_penalty > 1.0:
            for token_id in set(generated_ids[prompt_length:]):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= repetition_penalty
                else:
                    next_token_logits[token_id] *= repetition_penalty

        # Temperature
        if temperature > 0:
            next_token_logits = (
                next_token_logits / temperature
            )

        # Never generate structural tokens as normal text
        next_token_logits[bos_id] = float("-inf")
        next_token_logits[user_id] = float("-inf")
        next_token_logits[pad_id] = float("-inf")
        next_token_logits[assistant_id] = float("-inf")

        # Top-k sampling
        if top_k is not None:

            k = min(
                top_k,
                next_token_logits.size(-1)
            )

            top_values, top_indices = torch.topk(
                next_token_logits,
                k=k
            )

            filtered_logits = torch.full_like(
                next_token_logits,
                float("-inf")
            )

            filtered_logits[top_indices] = top_values

            next_token_logits = filtered_logits

        probabilities = F.softmax(
            next_token_logits,
            dim=-1
        )

        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        generated_ids.append(
            next_token_id
        )

        # End of assistant response
        if next_token_id == eos_id:
            break

    # Extract everything after <ASSISTANT>
    try:
        assistant_position = generated_ids.index(
            assistant_id
        )

        response_ids = generated_ids[
            assistant_position + 1:
        ]

    except ValueError:
        response_ids = generated_ids

    # Remove special tokens
    response_ids = [
        token_id
        for token_id in response_ids
        if token_id not in [
            bos_id,
            user_id,
            assistant_id,
            pad_id,
            eos_id
        ]
    ]

    return tokenizer.decode(
        response_ids
    ).strip()