"""Pipeline orchestration: pick a function, then fill each parameter."""
from typing import Any, Protocol

from .constraints import (
    Constraint, EnumConstraint, NumberConstraint, StringConstraint,
)
from .decoder import generate_constrained
from .models import FunctionCallResult, FunctionDefinition, TestCase
from .prompt_board import build_param_prompt, build_selection_prompt

NAME_MAX_TOKENS = 32
NUMBER_MAX_TOKENS = 16
STRING_MAX_TOKENS = 64


class LLMSDKProtocol(Protocol):
    """Protocol matching the required methods on Small_LLM_Model."""

    def encode(self, text: str) -> Any:
        """Tokenize ``text`` into a tensor of ids (batch dimension 0)."""
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for the given ids."""
        ...


def get_constraint_for_param(param_type: str) -> Constraint:
    """Return the Constraint matching a schema parameter type."""
    if param_type == "number":
        return NumberConstraint(allow_float=True, allow_negative=True)
    if param_type == "integer":
        return NumberConstraint(allow_float=False, allow_negative=True)
    if param_type == "boolean":
        return EnumConstraint(candidates=["true", "false"])
    return StringConstraint()


def parse_generated_value(raw_str: str, param_type: str) -> Any:
    """Turn the model's raw text into a plain Python value.

    The decoder already guarantees the text is valid for the type, so
    this is a plain conversion. ``number`` is always a float, matching
    the subject's example output (``"a": 2.0``); json.dump() handles all
    quoting and escaping later.
    """
    if param_type == "number":
        return float(raw_str)
    if param_type == "integer":
        return int(raw_str)
    if param_type == "boolean":
        return raw_str == "true"
    return raw_str


def _encode(sdk: LLMSDKProtocol, prompt: str) -> list[int]:
    """Tokenize a prompt into a plain list of ids."""
    ids: list[int] = sdk.encode(prompt)[0].tolist()
    return ids


def exe_func_call_pipline(
    sdk: LLMSDKProtocol,
    id_to_text: dict[int, str],
    eos_id: int,
    test_case: TestCase,
    funcs: list[FunctionDefinition],
    debug: bool = False,
) -> FunctionCallResult:
    """Process one prompt: choose a function, then fill its parameters.

    Each parameter is a separate constrained generation with its own
    fresh prompt, so earlier answers cannot leak into later ones.

    Args:
        sdk: The model wrapper.
        id_to_text: Normalized token id -> text map.
        eos_id: The end-of-turn token id (``<|im_end|>``).
        test_case: The natural-language request.
        funcs: Available function definitions.
        debug: Print per-step decoder diagnostics.
    """
    if debug:
        print(f"\n=== prompt: {test_case.prompt!r} ===")
    fn_map = {fn.name: fn for fn in funcs}

    # 1. Choose the function name, restricted to the known names.
    selected_fn_name = generate_constrained(
        sdk=sdk, id_to_text=id_to_text, eos_id=eos_id,
        input_ids=_encode(sdk, build_selection_prompt(test_case.prompt,
                                                      funcs)),
        constraint=EnumConstraint(candidates=list(fn_map)),
        max_new_tokens=NAME_MAX_TOKENS, debug=debug)
    target_fn = fn_map[selected_fn_name]

    # 2. Fill each parameter with its own prompt and constraint.
    parsed_params: dict[str, Any] = {}
    for param_name, param_def in target_fn.parameters.items():
        if debug:
            print(f"  [param {param_name!r} ({param_def.type})]")
        cap = (NUMBER_MAX_TOKENS if param_def.type in ("number", "integer")
               else STRING_MAX_TOKENS)
        raw_value = generate_constrained(
            sdk=sdk, id_to_text=id_to_text, eos_id=eos_id,
            input_ids=_encode(sdk, build_param_prompt(
                test_case.prompt, target_fn, param_name)),
            constraint=get_constraint_for_param(param_def.type),
            max_new_tokens=cap, debug=debug)
        # The prompt already ends with "<param>:", so the value starts
        # right after it: drop the one space that follows the colon.
        parsed_params[param_name] = parse_generated_value(
            raw_value.removeprefix(" "), param_def.type)

    return FunctionCallResult(
        prompt=test_case.prompt,
        name=selected_fn_name,
        parameters=parsed_params,
    )