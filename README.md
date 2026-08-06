# llm-delusion-eval

## Introduction

This is an eval designed to measure whether models encourage delusion-linked behaviors.

![An example DelusionEval window replay with original and evaluated model responses.](assets/img/delusioneval_window_cutup_real_data.png)

*An example of DelusionEval: we take an existing conversational window from a user's transcript and evaluate a model by successively prompting it with chains of the original context.*

DelusionEval is built from real user-assistant transcripts from 18 users who reported psychological harm from LLMs. After retaining user and assistant turns, we split conversations into overlapping windows of up to 20 messages and curated 677 code-conditioned conversation histories (589 unique histories), comprising 12,591 messages. For each sample, the evaluated model receives the exact transcript prefix up to a user turn and generates the next assistant reply, and the next sample resets to the original transcript rather than chaining on the evaluated model's previous output.

The DelusionEval dataset page is available at
[Hugging Face](https://huggingface.co/datasets/jlcmoore/delusioneval).

## Release Notes

- This repository is intended for a one-time release; external contributions are not accepted.
- The controlled-access data is gated under a data use agreement for non-commercial scientific research and may not be redistributed or used to re-identify participants.

For the full setup and evaluation reference see [EXTENDED_REFERENCE.md](EXTENDED_REFERENCE.md).

<!--  -->
<!--  -->
<!--  -->
<!--  -->
<!--  -->
<!--  -->

## Setup

### Installation

```sh
uv sync
```

### Setting up API keys for models

Inspect AI supports `.env` files, so the easiest way to manage API keys is to add them to the `.env` file. For example:

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=tgp_...
ANTHROPIC_API_KEY=sk-...
```

Inspect AI will automatically load the environment variables from this file on startup.

### Data Setup

Window-only evals use the public sanitized dataset hosted on Hugging Face:

```sh
export LLM_DELUSIONS_WINDOWS_PATH=hf://datasets/spiralsafety/delusioneval/items_sanitized.parquet
```

<!--  -->
<!--  -->
<!--  -->
<!--  -->
<!--  -->

## Evaluation

### Dry Run

If you want to verify that your data loads correctly and preview the exact prompts being sent to the models without incurring API costs, you can use Inspect's built-in `mockllm/model`.

However, because the `delusions_eval` task uses a grader that expects JSON output, the standard `mockllm/model` will fail with a `ClassificationError`. To avoid this, use the provided mock wrapper:

```sh
uv run python src/llm_delusion_eval/scripts/inspect_mocked.py eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model mockllm/model \
  --model-role grader=mockllm/model \
  --limit 2 \
  -T mock_score=7
```

### Examples

**Run all behaviors on the full sanitized dataset:**

```sh
MODEL=openai/gpt-5.4-mini-2026-03-17
GRADER_JSON='{"model":"openai/gpt-5.1-2025-11-13","reasoning_effort":"none"}'
uv run inspect eval src/llm_delusion_eval/tasks/delusions_eval.py \
  --model "$MODEL" \
  --model-role "grader=$GRADER_JSON"
```

## Viewing Results

```sh
uv run inspect view
```

Results are stored in the `logs/` directory (or `$INSPECT_LOG_DIR` if set). For analysis workflows and artifact outputs, see `analysis/Analysis_README.md`.


## Citation

Please cite this evaluation using the following paper citation:

```bibtex
@misc{moore_delusioneval_2026,
  title = {DelusionEval: Measuring Delusion-Linked Behaviors in AI Chatbots},
  author = {Moore, Jared and Mock, Andrea and Mai, Yifan and Anthis, Jacy Reese and Louie, Ryan and Agnew, William and Mehta, Ashish and Klyman, Kevin and Liang, Percy and Haber, Nick and Lin, Eric and Ong, Desmond C.},
  year = {2026},
  url = {https://arxiv.org/abs/2608.05004},
  note = {Preprint},
}
```