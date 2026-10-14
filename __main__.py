"""Entry point for the function-calling tool.

Run with: uv run python -m src [--functions_definition FILE] [--input FILE] [--output FILE]
"""

import argparse
import json
import time

from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition, TestCase
from .heist_plan import exe_func_call_pipline
from .decoder import build_id_to_text
from .prompt_board import IM_END

DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tool."""
    parser = argparse.ArgumentParser(description="Function calling via constrained decoding.")
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS_DEFINITION)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--debug", action="store_true",
        help="Print raw top-5 candidates and the masked winner at "
             "every generation step (also DEBUG_DECODE=1 env var).")
    return parser.parse_args()


def main() -> None:
    """Entry point: load model + inputs, run the pipeline, write results."""
    args = parse_args()
    start_total_time = time.perf_counter()
    print("Loading model...")
    start_model_load = time.perf_counter()
    sdk = Small_LLM_Model()
    model_loading_dur = time.perf_counter() - start_model_load
    print(f"Model loaded in {model_loading_dur}.\n")
    # ids = sdk.encode("<|im_start|>user")
    # print("new check:")
    # print(len(ids[0].tolist()), sdk.decode(ids[0].tolist()))
    # print("end check")
    # loading vocab mapping but in reverse
    vocab_path = sdk.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    id_to_token = {token_id: token_str for token_str, token_id in raw_vocab.items()}
    id_to_text = build_id_to_text(id_to_token)
    eos_ids = sdk.encode(IM_END)[0].tolist()
    if len(eos_ids) != 1:
        print(f"Error: {IM_END} should be one token, got {eos_ids}")
        return
    eos_id = eos_ids[0]
    # loading func schema
    try:
        with open(args.functions_definition, "r", encoding="utf-8") as f:
            raw_funcs = json.load(f)
        funcs = [FunctionDefinition(**fn) for fn in raw_funcs]
    except FileNotFoundError:
        print(f"Error: functions definition file not found: {args.functions_definition}")
        return
    except json.JSONDecodeError as jde:
        print(f"Error: failed to parse JSON functions file {args.functions_definition}: {jde}")
        return

    # loading test cases
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            raw_teste = json.load(f)
            test_cases = [TestCase(**t_c) for t_c in raw_teste]
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input}")
        return
    except json.JSONDecodeError as jde:
        print(f"Error: Failed to parse JSON input file {args.input}: {jde}")
        return

    print(f"Running generation for tot amount of {len(test_cases)} test prompts...")
    rexes = []
    strt_gens = time.perf_counter()
    for test_case in test_cases:
        print(f"processand:{test_case.prompt!r}")
        try:
            # rex = exe_func_call_pipline(sdk=sdk,
            #     id_to_token=id_to_token, test_case=test_case,
            #     funcs=funcs, debug=args.debug)
            rex = exe_func_call_pipline(sdk=sdk, 
                id_to_text=id_to_text, eos_id=eos_id,
                test_case=test_case, funcs=funcs, debug=args.debug)
            rexes.append(rex.model_dump())

            # #checking stuff
            # im_end_ids = sdk.encode("<|im_end|>")[0].tolist()
            # print(f"witnessing the end: ",im_end_ids, im_end_ids[0] in id_to_token)

        except Exception as exc:
            print(f"Warning: skipping prompt {test_case.prompt!r}: {exc}")
    gen_dur = time.perf_counter() - strt_gens
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rexes, f, indent=2)
    tot_duration = (time.perf_counter() - start_total_time) / 60
    print(f"\nDone! GG eazyjob!\n saved {len(rexes)} results to {args.output}")
    print(f"Generation time: {gen_dur:.2f}s", f"Total script runtime: {tot_duration:.2f} min")


if __name__ == "__main__":
    main()
