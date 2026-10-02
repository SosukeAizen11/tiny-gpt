import re


class Tokenizer:

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text: str) -> list[str]:
        """
        Convert text into tokens.

        Special tokens such as <USER> and <ASSISTANT>
        are preserved as single tokens. Numbers, currency amounts,
        emails, phone numbers, and contractions are captured atomically.
        """

        special_pattern = (
            r"<USER>|<ASSISTANT>|<PAD>|"
            r"<UNK>|<BOS>|<EOS>"
        )

        currency_pattern = r"\$\d+(?:,\d+)*(?:\.\d+)?"
        decimal_pattern = r"\b\d+\.\d+\b"
        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
        phone_pattern = r"\b(?:\d{1,4}-)?\d{3}-\d{3,4}\b"
        contraction_pattern = r"\b\w+['’]\w+\b"

        pattern = (
            special_pattern +
            r"|" + email_pattern +
            r"|" + phone_pattern +
            r"|" + currency_pattern +
            r"|" + decimal_pattern +
            r"|" + contraction_pattern +
            r"|\w+|[^\w\s]"
        )

        return re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

    def build_vocabulary(
        self,
        texts: list[str]
    ) -> None:

        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            "<USER>",
            "<ASSISTANT>",
        ]

        unique_tokens = set()

        for text in texts:

            tokens = self.tokenize(text)

            unique_tokens.update(tokens)

        # Don't duplicate special tokens
        unique_tokens -= set(special_tokens)

        vocabulary = (
            special_tokens
            + sorted(unique_tokens)
        )

        self.token_to_id = {
            token: index
            for index, token in enumerate(
                vocabulary
            )
        }

        self.id_to_token = {
            index: token
            for token, index
            in self.token_to_id.items()
        }

    def encode(
        self,
        text: str
    ) -> list[int]:

        tokens = self.tokenize(text)

        unk_id = self.token_to_id[
            "<UNK>"
        ]

        return [
            self.token_to_id.get(
                token,
                unk_id
            )
            for token in tokens
        ]

    def decode(
        self,
        token_ids: list[int]
    ) -> str:

        tokens = [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

        text = " ".join(tokens)
        # Clean up spaces before punctuation
        text = re.sub(r"\s+([.,!?:;])", r"\1", text)
        # Clean up spaces inside hyphens, colons in time, and apostrophes
        text = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", text)
        text = re.sub(r"(\d)\s*:\s*(\d)", r"\1:\2", text)
        text = re.sub(r"(\w)\s*'\s*(\w)", r"\1'\2", text)
        text = re.sub(r"\$\s+(\d)", r"$\1", text)
        return text

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)