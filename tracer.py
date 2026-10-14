import re
from typing import Protocol

# 1. Dummy interface mimicking NumberConstraint & Decoder logic
class NumberConstraint:
    def __init__(self, allow_float: bool = True):
        self._allow_float = allow_float

    def valid_continuation(self, so_far: str, candidate_token: str) -> bool:
        extended = so_far + candidate_token
        stripped = extended.strip()
        if not stripped or stripped == "-":
            return True
        pattern = r"^-?\d+(\.\d*)?$" if self._allow_float else r"^-?\d+$"
        return bool(re.match(pattern, stripped))

    def is_complete(self, so_far: str) -> bool:
        has_digits = bool(re.search(r"\d", so_far))
        has_delimiter = so_far.endswith((" ", "\n", "\t", ",", '"', "'", "<|im_end|>"))
        return has_digits and has_delimiter


# 2. Simulated vocabulary (showing BPE token representations)
simulated_vocab = {
    101: "2",
    102: "265",
    103: "65",
    104: "345",
    105: " ",
    106: "hello",
    107: "3.14",
    108: "<|im_end|>",
}

# Simulated logits from model for prompt: "What is the sum of 265 and 345?"
# Notice token 102 ('265') has a higher logit score than 101 ('2')
simulated_logits = {
    101: 5.2,   # "2"
    102: 12.8,  # "265"  <-- Model wants to output full token!
    103: 1.1,   # "65"
    104: 3.4,   # "345"
    105: 8.5,   # " "
    106: 15.0,  # "hello" (High score, but masked to -inf!)
    107: 0.2,   # "3.14"
    108: 2.0    # "<|im_end|>"
}


def trace_constrained_generation():
    constraint = NumberConstraint()
    generated_text = ""
    print("=== STARTING CONSTRAINED GENERATION TRACE ===\n")

    for step in range(1, 5):
        print(f"--- Step {step} ---")
        print(f"Current `so_far` string: {generated_text!r}")
        best_id = None
        best_score = float("-inf")
        best_token_str = ""
        # Masking loop
        for token_id, token_str in simulated_vocab.items():
            raw_logit = simulated_logits[token_id]
            # Check valid continuation
            is_valid = constraint.valid_continuation(generated_text, token_str)
            if not is_valid:
                masked_logit = float("-inf")
                status = "MASKED (-inf)"
            else:
                masked_logit = raw_logit
                status = f"ALLOWED (logit={masked_logit})"
                if masked_logit > best_score:
                    best_score = masked_logit
                    best_id = token_id
                    best_token_str = token_str
            print(f"  Token {token_id} ({token_str!r:10s}): {status}")
        if best_id is None:
            print("\n[!] No valid tokens found. Stopping.")
            break

        generated_text += best_token_str
        print(f"\n-> Selected Token: {best_token_str!r} (ID: {best_id})")
        print(f"-> Updated `generated_text`: {generated_text!r}")

        # Completion Check
        is_done = constraint.is_complete(generated_text)
        print(f"-> `is_complete()` evaluation: {is_done}\n")

        if is_done:
            print("=== GENERATION COMPLETE ===")
            break

    print(f"Final Parsed Number: {generated_text.strip()!r}")

if __name__ == "__main__":
    trace_constrained_generation()