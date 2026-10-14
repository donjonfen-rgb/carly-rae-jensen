"""Prompt construction for Qwen3 (ChatML) function calling.

Everything the model *sees* is built here, so heist_plan.py only has to
orchestrate. The layout mirrors Qwen3's chat template with thinking off:

    <|im_start|>role\\ncontent<|im_end|>\\n   (one block per message)
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n   (generation here)

The prompt ends exactly there: no "{" or other priming text. The
constraint decides what may be emitted; the model ends its answer with
<|im_end|>, which the decoder treats as the stop signal.
"""
from typing import Literal

from pydantic import BaseModel

from .models import FunctionDefinition, ParameterDefinition

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# Qwen3's way to switch thinking off: a pre-closed(?), empty block.
EMPTY_THINK = "<think>\n\n</think>\n\n"


class Message(BaseModel):
    """A single chat turn."""

    role: Literal["system", "user", "assistant"]
    content: str


def render_chatml(messages: list[Message],
                  assistant_prefix: str = "") -> str:
    """Render messages as a ChatML string ready for tokenization.

    Args:
        messages: The conversation so far, ending with the user turn
            that should be answered.
        assistant_prefix: Text already written for the assistant. The
            model continues right after it, so it cannot repeat it.

    Returns:
        The prompt text, ending at the start of the assistant's answer
        with thinking already closed.
    """
    parts = [f"{IM_START}{m.role}\n{m.content}{IM_END}\n" for m in messages]
    parts.append(f"{IM_START}assistant\n{EMPTY_THINK}{assistant_prefix}")
    return "".join(parts)


def describe_function(fn: FunctionDefinition) -> str:
    """Return a compact, human-readable description of one function."""
    lines = [f"- {fn.name}: {fn.description}"]
    for name, param in fn.parameters.items():
        extra = f" - {param.description}" if param.description else ""
        lines.append(f"    * {name} ({param.type}){extra}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Step 1: choosing the function
# --------------------------------------------------------------------------
_SELECT_SYSTEM = (
    "You route user requests to functions. Given the list of available "
    "functions, reply with ONLY the name of the single function that "
    "best fulfils the request. Do not explain."
)


def build_selection_prompt(query: str, funcs: list[FunctionDefinition]) -> str:
    """Build the prompt that asks the model to pick a function name."""
    catalog = "\n".join(describe_function(fn) for fn in funcs)
    user = f"Available functions:\n{catalog}\n\nRequest: {query}"
    return render_chatml([
        Message(role="system", content=_SELECT_SYSTEM),
        Message(role="user", content=user),
    ])


# --------------------------------------------------------------------------
# Step 2: extracting one parameter with full context
# --------------------------------------------------------------------------
# No concrete pattern examples here on purpose: a 0.6B model copies any
# example it is shown, so the rules stay abstract and the few-shot demos
# below cover several different kinds of answer instead.
_PARAM_SYSTEM = (
    "You fill in ONE argument of a function call from a user request. "
    "Reply in the form '<parameter name>: <value>' where the value is "
    "the raw value of the requested parameter: no quotes, no explanation.\n"
    "- number: digits only (write spelled-out numbers as digits).\n"
    "- string: the text the function should receive, copied COMPLETELY "
    "from the request. When it is quoted, copy everything up to the "
    "closing quote and never shorten it.\n"
    "- pattern or regular expression: write the regular expression that "
    "matches exactly what the request asks to find. A specific word "
    "stays a literal word; a described class of characters becomes a "
    "character class.\n"
    "- replacement text is written literally."
)

# Fictional functions: they demonstrate the answer FORMAT only. The demo
# answers deliberately differ from each other (literal word, character
# class) so the model learns to adapt rather than to copy one pattern.
_DEMO_REPEAT = FunctionDefinition(
    name="fn_repeat_text",
    description="Repeat a piece of text a given number of times.",
    parameters={
        "text": ParameterDefinition(type="string"),
        "times": ParameterDefinition(type="number"),
    },
)
_DEMO_DELETE = FunctionDefinition(
    name="fn_delete_matches",
    description="Delete every match of a regular expression from a string.",
    parameters={
        "source_string": ParameterDefinition(type="string"),
        "pattern": ParameterDefinition(type="string"),
    },
)
_DEMOS: list[tuple[FunctionDefinition, str, str, str]] = [
    (_DEMO_REPEAT, "Say 'hi there' three times", "text", "hi there"),
    (_DEMO_REPEAT, "Say 'hi there' three times", "times", "3"),
    # Two quoted spans: the model must tell the short word from the full
    # text and copy the full text to its closing quote.
    (_DEMO_DELETE, "Delete the word 'blue' from 'The blue car and the "
     "blue bike'", "source_string", "The blue car and the blue bike"),
    (_DEMO_DELETE, "Delete the word 'blue' from 'The blue car and the "
     "blue bike'", "pattern", "blue"),
    (_DEMO_DELETE, "Strip every uppercase letter from 'Hello World'",
     "pattern", "[A-Z]"),
]


def _param_user_message(fn: FunctionDefinition, query: str,
                        param_name: str) -> str:
    """Format the user turn asking for one parameter of ``fn``."""
    param = fn.parameters[param_name]
    return (f"Function:\n{describe_function(fn)}\n\n"
            f"Request: {query}\n"
            f"Parameter to fill: {param_name} ({param.type})")


def build_param_prompt(query: str, fn: FunctionDefinition,
                       param_name: str) -> str:
    """Build the prompt that extracts a single parameter value.

    The model sees the chosen function's name, description and all its
    parameters (not just the target one), plus demo turns showing the
    exact answer format.
    """
    messages = [Message(role="system", content=_PARAM_SYSTEM)]
    for demo_fn, demo_query, demo_param, demo_answer in _DEMOS:
        messages.append(Message(
            role="user",
            content=_param_user_message(demo_fn, demo_query, demo_param)))
        messages.append(Message(
            role="assistant", content=f"{demo_param}: {demo_answer}"))
    messages.append(Message(
        role="user", content=_param_user_message(fn, query, param_name)))
    # The label is already written for the model ("name:"), so all it
    # can do is write the value after it.
    return render_chatml(messages, assistant_prefix=f"{param_name}:")