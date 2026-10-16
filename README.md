# Prompt masking on agentic trajectories

This repository is a small, reproducible pilot for comparing assistant-only
supervision with full prompt-and-assistant loss on tool-using agent trajectories.
It adapts the experiment discipline in
[`euiyulsong/turn-loss-benchmark`](https://github.com/euiyulsong/turn-loss-benchmark):
same base model, same examples and target answers, paired evaluation, explicit
mask checks, and conservative interpretation.

## Status

The first implementation targets **ToolBench** through a normalized JSONL
adapter. τ-bench and AgentBench are listed as planned adapters, not yet
implemented or validated; their task environments and data access requirements
make them separate integrations, not interchangeable training corpora. Do not
claim cross-benchmark results until each adapter has been audited and run.

The `toy` smoke-test dataset is synthetic and tests pipeline mechanics only; it
is not evidence about agent quality.

A larger ToolBench pilot (512 train / 500 held-out; 200 updates per objective)
has been completed. Its limited results and protocol deviations are documented
in [`REPORT.md`](REPORT.md), with summary metrics in
[`RESULTS/toolbench-large-summary.json`](RESULTS/toolbench-large-summary.json).
The held-out lexical metrics compare generated trajectories, not verified task
success. Final-answer subspan EM has been extracted for the existing 100
predictions; because generation was capped at 64 tokens, neither model reached a
Finish/final-answer span in that slice.

## Research question and conditions

Primary comparison: train identical trajectories with (A) assistant-message
tokens supervised and (B) every message supervised, including user/tool
observations. Both conditions use the same examples, order, seed, optimizer
steps, base checkpoint, and evaluation tasks.

Full loss changes both prompt-token supervision and assistant-token relative
weight. Therefore this estimates the effect of two practical objectives, not
the isolated causal effect of adding prompt loss. A follow-up can use
`mean(assistant CE) + lambda * mean(prompt CE)` with a declared lambda sweep.

Assistant spans include role header, body, and end-of-message marker. User and
tool messages are masked in assistant-only mode. Template boundaries are
validated; this is not string concatenation and not sequential per-turn
optimization.

## Quick start

Requires Python 3.11+, a recent PyTorch/Transformers stack, and a supported
chat-template causal LM. GPU is strongly recommended. Start with the offline
smoke test:

```bash
python -m pip install -r requirements.txt
python trajectory_bench.py smoke
```

No benchmark data, model weights, credentials, or generated trajectories are
committed. The initial ToolBench adapter requires locally downloaded
`toolllama_G123_dfs_train.json` and `toolllama_G123_dfs_eval.json` JSON files.
τ-bench may require live domain simulators; AgentBench environments have
distinct setup and license requirements.

## Protocol requirements

- Split by task/conversation identity before rendering trajectories; prevent
  near-duplicate or shared-task leakage across train/eval.
- Keep held-out tasks identical across conditions. Report task-level paired
  bootstrap intervals, not token-level intervals.
- Evaluate actual agent outcomes (task success / benchmark-native metrics),
  tool-call validity, and turn-level failures. Do not score only assistant text.
- Record model and tokenizer revisions, dataset/environment version, hashes,
  seeds, token counts, loss-token counts, optimizer steps, runtime, and hardware.
- Report seed variation separately from task bootstrap uncertainty. A single
  seed is a pilot only.
- Never place API secrets in configs, logs, result files, or commits.

See [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the preregistered pilot
matrix and [`docs/DATASETS.md`](docs/DATASETS.md) for adapter status.

## Attribution

The experimental framing is informed by the linked turn-loss benchmark. This
repository is an independent implementation; it does not include their source
data or model artifacts.
