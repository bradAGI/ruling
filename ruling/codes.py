"""Single-token answer codes, verified against the tokenizer at load time."""

from __future__ import annotations

from dataclasses import dataclass

CANDIDATES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class AnswerCodes:
    codes: list[str]
    token_ids: list[int]

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "AnswerCodes":
        codes: list[str] = []
        ids: list[int] = []
        for code in CANDIDATES:
            tokens = tokenizer.encode(code, add_special_tokens=False)
            if len(tokens) == 1:
                codes.append(code)
                ids.append(tokens[0])
        if len(codes) < 2:
            raise ValueError("tokenizer yields fewer than two single-token answer codes")
        return cls(codes=codes, token_ids=ids)

    @property
    def max_options(self) -> int:
        return len(self.codes)
