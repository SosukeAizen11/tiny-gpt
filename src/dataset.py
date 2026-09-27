import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):

    def __init__(
        self,
        texts,
        tokenizer,
        context_length
    ):
        self.inputs = []
        self.targets = []

        bos_id = tokenizer.token_to_id["<BOS>"]
        eos_id = tokenizer.token_to_id["<EOS>"]
        pad_id = tokenizer.token_to_id["<PAD>"]

        # Process every conversation separately
        for text in texts:

            token_ids = tokenizer.encode(text)

            sequence = (
                [bos_id]
                + token_ids
                + [eos_id]
            )

            # --------------------------------
            # Short conversation
            # --------------------------------

            if len(sequence) < context_length + 1:

                input_sequence = sequence[:-1]
                target_sequence = sequence[1:]

                # Pad input
                input_sequence = (
                    input_sequence
                    + [pad_id] * (
                        context_length
                        - len(input_sequence)
                    )
                )

                # Pad target
                target_sequence = (
                    target_sequence
                    + [pad_id] * (
                        context_length
                        - len(target_sequence)
                    )
                )

                self.inputs.append(
                    torch.tensor(
                        input_sequence,
                        dtype=torch.long
                    )
                )

                self.targets.append(
                    torch.tensor(
                        target_sequence,
                        dtype=torch.long
                    )
                )

                continue

            # --------------------------------
            # Long conversation
            # --------------------------------

            for i in range(
                len(sequence) - context_length
            ):

                input_sequence = sequence[
                    i:i + context_length
                ]

                target_sequence = sequence[
                    i + 1:i + context_length + 1
                ]

                self.inputs.append(
                    torch.tensor(
                        input_sequence,
                        dtype=torch.long
                    )
                )

                self.targets.append(
                    torch.tensor(
                        target_sequence,
                        dtype=torch.long
                    )
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return (
            self.inputs[index],
            self.targets[index]
        )