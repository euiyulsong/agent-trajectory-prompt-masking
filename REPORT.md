# ToolBench prompt-masking pilot

## Status and scope

This is a **single-seed, short pilot**, not a benchmark-quality conclusion. It
compares assistant-only masking against full-token loss on a fixed subset from
ToolBench. No τ-bench or AgentBench experiment has been run.

## Setup

- Source: `Yhyu13/ToolBench_toolllama_G123_dfs`, Apache-2.0 dataset card; it
  repackages ToolBench conversations.
- Base: `Qwen/Qwen2.5-1.5B-Instruct` (Hub default revision; revision SHA was not
  pinned, so exact model artifact reproducibility is incomplete).
- Hardware: NVIDIA GeForce RTX 4090, 24 GB.
- Software: Python 3.10, PyTorch 2.5.1+cu124, Transformers 4.51.3, PEFT 0.15.2,
  Datasets 3.5.0, Accelerate 1.6.0.
- Seed: 42; LoRA r=8, alpha=16; 30 optimizer updates per condition; batch 1,
  gradient accumulation 2; max sequence length 2048.
- Data: 57 train trajectories and 27 held-out-from-training trajectories from
  the source train split. The source's separate 762-row test split was not used.
  Selection was deterministic after seed-42 shuffle. Twelve selected examples
  were excluded for length. The original static system prompt contained very
  large repeated tool schemas; the pilot replaces that system prompt with the
  first user task while preserving the ordered user/assistant/tool trajectory.
  This is a documented format deviation and limits claims to this pilot setup.
- Assistant spans were constructed by chat-template prefix differencing. The
  Qwen template emitted no assistant mask, so template prefix stability was
  checked and assistant message headers/body/terminators were retained.

## Results

| Objective | Train loss | Supervised tokens/epoch | Mean held-out assistant NLL | Whole-trajectory token-overlap F1 | Exact whole-response match |
| --- | ---: | ---: | ---: | ---: | ---: |
| Assistant-only | 1.023 | 20,887 | **0.9680** | **0.392** | 0 / 27 |
| Full prompt + assistant | 1.415 | 50,074 | 0.9877 | 0.336 | 0 / 27 |

Both conditions used the same 50,074 input tokens, 57 examples, seed, and 30
updates. Evaluation is teacher-forced NLL on the final assistant response for
27 fixed examples plus deterministic generation capped at 128 tokens. Exact
match compares the entire generated trajectory string (reasoning, tool syntax,
and final answer) with the reference and is therefore too strict to be an
informative quality measure. Token-overlap F1 is also over the entire trajectory
and is only a rough lexical diagnostic, not task success. No ToolBench
task-success score or paired confidence interval is reported. Better follow-up
evaluation should parse and execute the predicted tool calls or score the final
answer separately.

Observed assistant-NLL difference (full minus assistant-only) is +0.0197 nats
per token in favor of assistant-only on this small held-out subset. This is
descriptive only: one seed, 27 tasks, short training, and system-prompt
compaction prevent a causal or general benchmark claim. Full loss also changes
assistant-token relative weight, so this is not an isolated prompt-loss effect.

## Artifacts

Adapters and generated predictions are local under ignored `runs/` and
`results/`; raw dataset files are not included. Run metadata is in each adapter's
`run.json`. The local `data/toolbench-pilot/manifest.json` records sample counts
and split details.

## Follow-up required

Pin model and dataset revisions; preserve the upstream tool schema using a
compact, equivalent representation; evaluate the official ToolBench test
protocol or a local deterministic tool simulator with task success; run at
least three seeds; compute paired task bootstrap intervals; and implement and
validate τ-bench and AgentBench adapters separately before making cross-dataset
claims.
