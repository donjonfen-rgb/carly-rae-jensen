"""Quick dependency/environment sanity check.

Not part of the graded submission -- run this any time you're unsure
whether the venv is actually set up correctly, without waiting for the
full model download to find out the hard way.

Run with: uv run python3 check_env.py
"""

import importlib
import sys

REQUIRED_PACKAGES = ["torch", "transformers", "huggingface_hub", "numpy", "pydantic"]


def main() -> None:
    print(f"Python: {sys.version}\n")

    all_ok = True
    for package_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  OK    {package_name:<20} {version}")
        except ImportError as exc:
            all_ok = False
            print(f"  FAIL  {package_name:<20} {exc}")

    print()
    try:
        from llm_sdk import Small_LLM_Model  # noqa: F401
        print("  OK    llm_sdk import successful (model not loaded yet)")
    except ImportError as exc:
        all_ok = False
        print(f"  FAIL  llm_sdk import failed: {exc}")

    print()
    if all_ok:
        print("All core dependencies import correctly.")
    else:
        print("Something is missing -- see FAIL lines above.")


if __name__ == "__main__":
    main()
