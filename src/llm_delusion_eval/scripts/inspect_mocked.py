"""
Wrapper for the `inspect` CLI that monkey-patches `mockllm/model`.

This allows `mockllm/model` to return a valid JSON response so that it
satisfies the requirements of the annotation grader during local testing.
You can use this script exactly as you would use the `inspect` CLI.

Usage:
    uv run python src/llm_delusion_eval/scripts/inspect_mocked.py eval \
    src/llm_delusion_eval/tasks/delusions_eval.py --model mockllm/model \
    --model-role grader=mockllm/model
"""

import sys

from inspect_ai._cli.main import main
from inspect_ai.model._providers.mockllm import MockLLM

DEFAULT_MOCK_SCORE = 5


def run():
    """
    Main entry point for the mocked inspect CLI. Parses arguments and runs
    the inspect CLI.
    """
    # 1. Extract a custom mock score from task arguments if present
    # Format: -T mock_score=7 or --task-arg mock_score=7
    mock_score = DEFAULT_MOCK_SCORE
    args_to_remove = []

    for i, arg in enumerate(sys.argv):
        if arg.startswith("mock_score="):  # handles -T mock_score=X
            try:
                mock_score = int(arg.split("=")[1])
                args_to_remove.append(i)
            except (ValueError, IndexError):
                pass

    # Clean up sys.argv so inspect doesn't complain about unknown task args
    for index in sorted(args_to_remove, reverse=True):
        # Remove the mock_score=N part
        sys.argv.pop(index)

        # If the previous arg was -T or --task-arg, and it's now "trailing",
        # we should remove it too to keep the CLI command valid.
        if index > 0 and sys.argv[index - 1] in ["-T", "--task-arg"]:
            # Check if there are other args in the same -T block (commas)
            # but for simplicity, if we used -T mock_score=N, we just remove the -T
            sys.argv.pop(index - 1)

    # 2. Spoof the mock model's default output to be valid JSON
    # This prevents the ClassificationError when mockllm/model acts as the grader
    MockLLM.default_output = (
        f'{{"score": {mock_score}, '
        f'"rationale": "mocking grader with score {mock_score}", '
        f'"quotes": [{{"quote": "dummy quote", '
        f'"reason": "mocking grader with score {mock_score}"}}]}}'
    )

    # 3. Hand off execution entirely to the standard inspect CLI
    sys.exit(main())


if __name__ == "__main__":
    run()
