"""Token-by-token constrained generation loop.

At every step the model scores all tokens. Tokens the constraint rejects
are skipped (equivalent to a logit of -inf) and the best remaining one is
chosen greedily.

Stopping works through the model's own end-of-turn token (``eos_id``,
``<|im_end|>`` for Qwen). That token is absent from vocab.json, so it is
handled separately: it competes with the regular tokens ONLY while the
constraint says the text so far is a complete value. If it wins, the
value is finished and the token is not added to the text.

Set DEBUG_DECODE=1 (or pass debug=True) to print, at every step, the true
top-5 tokens BEFORE masking, including ``<EOS>`` and other special tokens
that vocab.json does not list, followed by the masked winner.
"""
import heapq
import os
from typing import Protocol

from .constraints import Constraint

DEBUG = os.environ.get("DEBUG_DECODE", "") == "1"


class LogitsProvider(Protocol):
    """Anything that can turn token ids into next-token logits."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return one logit per vocabulary entry for the next token."""
        ...


def normalize_token(token_text: str) -> str:
    """Convert a raw vocab token into plain text.

    Byte-level BPE writes a space as 'Ġ' (U+0120) and a newline as 'Ċ'
    (U+010A); both are mapped back so constraints see real characters.
    """
    return token_text.replace("\u0120", " ").replace("\u010a", "\n")


def build_id_to_text(id_to_token: dict[int, str]) -> dict[int, str]:
    """Normalize every vocab entry once, instead of once per step."""
    return {i: normalize_token(t) for i, t in id_to_token.items()}


def _label(token_id: int, id_to_text: dict[int, str], eos_id: int) -> str:
    """Readable name for a token id, for debug output."""
    if token_id == eos_id:
        return "<EOS>"
    return repr(id_to_text.get(token_id, f"<special {token_id}>"))


def generate_constrained(sdk: LogitsProvider, id_to_text: dict[int, str],
    input_ids: list[int],constraint: Constraint, eos_id: int,
    max_new_tokens: int = 64, debug: bool = DEBUG) -> str:
    """Generate one constrained value.
    Args:
        sdk: Provides next-token logits.
        id_to_text: Token id -> normalized text (see build_id_to_text).
        input_ids: The tokenized prompt.
        constraint: Decides which tokens are legal and when to stop.
        eos_id: The end-of-turn token id the model uses to finish.
        max_new_tokens: Safety cap on generated tokens.
        debug: Print per-step diagnostics.
    Returns:
        The generated text, without any terminator.
    Raises:
        RuntimeError: If no legal token exists, or the model does not
            finish a valid value within ``max_new_tokens``.
    """
    text = ""
    working_ids = list(input_ids) # never mutate the caller's list
    keep_going = constraint.keep_going

    for step in range(max_new_tokens):
        logits = sdk.get_logits_from_input_ids(working_ids)

        if debug:
            top = heapq.nlargest(5, range(len(logits)),
                                 key=logits.__getitem__)
            shown = ", ".join(
                f"{_label(i, id_to_text, eos_id)}={logits[i]:.1f}"
                for i in top)
            print(f"    step {step:02d} RAW top-5 (pre-mask): {shown}")

        best_id: int | None = None
        best_score = float("-inf")
        if constraint.can_stop(text):
            best_id, best_score = eos_id, logits[eos_id]

        for token_id, token_text in id_to_text.items():
            if token_id == eos_id or not keep_going(text, token_text):
                continue
            score = logits[token_id]
            if score > best_score:
                best_id, best_score = token_id, score

        if best_id is None:
            raise RuntimeError(
                f"No legal token after {text!r}: the constraint cannot "
                "be satisfied from this state.")

        if debug:
            label = _label(best_id, id_to_text, eos_id)
            print(f"    step {step:02d}: picked {label} "
                  f"(id={best_id}, score={best_score:.2f}) | so_far={text!r}")

        if best_id == eos_id:
            return text
        working_ids.append(best_id)
        text += id_to_text[best_id]

    raise RuntimeError(f"Value not finished within {max_new_tokens}"
                       f" tokens: {text!r}")