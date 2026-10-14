"""Constraints that decide which tokens may be generated for one value.

A constraint answers two questions about the text generated so far:

* ``keep_going(so_far, token)``: may this token legally come next?
* ``can_stop(so_far)``: is ``so_far`` already a complete, valid value?

Ending a value is NOT done with marker text. The decoder lets the model
emit its end-of-turn token (``<|im_end|>``), but only while ``can_stop``
is true. The generated text therefore never contains terminators and
nothing has to be stripped afterwards.
"""
import re
import string
from typing import Protocol

# Printable ASCII on a single line: letters, digits, punctuation, space.
# Line breaks, tabs and other control characters are excluded, and so is
# every non-ASCII byte-level BPE fragment (the old "mojibake" source).
_VALUE_CHARS = frozenset(string.printable) - set("\n\r\t\x0b\x0c")


class Constraint(Protocol):
    """Interface every constraint type implements."""

    def keep_going(self, so_far: str, token: str) -> bool:
        """Return True if ``token`` may legally follow ``so_far``."""
        ...

    def can_stop(self, so_far: str) -> bool:
        """Return True if ``so_far`` is a complete, valid value."""
        ...


class EnumConstraint:
    """Output must be exactly one of a fixed set of candidate strings."""

    def __init__(self, candidates: list[str]) -> None:
        """Store the allowed strings (at least one is required)."""
        if not candidates:
            raise ValueError("EnumConstraint needs at least one candidate")
        self._candidates = list(candidates)

    def keep_going(self, so_far: str, token: str) -> bool:
        """Allow tokens that keep ``so_far`` a prefix of a candidate."""
        if not token:
            return False
        extended = so_far + token
        return any(c.startswith(extended) for c in self._candidates)

    def can_stop(self, so_far: str) -> bool:
        """Stopping is legal once ``so_far`` equals a whole candidate."""
        return so_far in self._candidates


class StringConstraint:
    """Free single-line ASCII text.

    json.dump() quotes and escapes the final Python string when the
    results file is written, so the model never produces JSON syntax.
    """

    def keep_going(self, so_far: str, token: str) -> bool:
        """Allow any non-empty token made of single-line ASCII."""
        return bool(token) and all(ch in _VALUE_CHARS for ch in token)

    def can_stop(self, so_far: str) -> bool:
        """Any string, including the empty one, is a valid value."""
        return True


class NumberConstraint:
    """A decimal number: optional sign, digits, optional fraction.

    One optional leading space is accepted, because the model writes
    "label: 265" as the tokens ":", " ", "2", "6", "5".
    """

    def __init__(self, allow_float: bool = True,
                 allow_negative: bool = True) -> None:
        """Compile the prefix and full-match patterns for this number."""
        sign = "-?" if allow_negative else ""
        if allow_float:
            self._partial = re.compile(rf" ?{sign}(\d+(\.\d*)?)?")
            self._complete = re.compile(rf" ?{sign}\d+(\.\d+)?")
        else:
            self._partial = re.compile(rf" ?{sign}\d*")
            self._complete = re.compile(rf" ?{sign}\d+")

    def keep_going(self, so_far: str, token: str) -> bool:
        """Allow tokens that keep ``so_far`` a valid number prefix."""
        if not token:
            return False
        return self._partial.fullmatch(so_far + token) is not None

    def can_stop(self, so_far: str) -> bool:
        """Stopping is legal once ``so_far`` is a complete number."""
        return self._complete.fullmatch(so_far) is not None