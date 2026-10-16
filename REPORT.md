# ToolBench prompt-masking pilot

## Status and scope

This is a **single-seed pilot**, not a benchmark-quality conclusion. It compares
assistant-only masking against full-token loss on fixed ToolBench subsets. No
τ-bench or AgentBench experiment has been run.

## Setup

- Source: `Yhyu13/ToolBench_toolllama_G123_dfs`, Apache-2.0 dataset card; it
  repackages ToolBench conversations.
- Base: `Qwen/Qwen2.5-1.5B-Instruct` (Hub default revision; revision SHA was not
  pinned, so exact model artifact reproducibility is incomplete).
- Hardware: NVIDIA GeForce RTX 4090, 24 GB.
- Software: Python 3.10, PyTorch 2.5.1+cu124, Transformers 4.51.3, PEFT 0.15.2,
  Datasets 3.5.0, Accelerate 1.6.0.
- Seed: 42; LoRA r=8, alpha=16; 200 optimizer updates per condition; batch 1,
  gradient accumulation 4; max sequence length 2048.
- Data: 512 train trajectories and a deterministic, task-disjoint 500-example
  held-out sample from the source train pool. All retained examples contain a
  tool observation; long examples over 2048 tokens are excluded. The source's
  separate 762-row test split was not used. The original static system prompt
  contains very large tool schemas; the experiment replaces it with a compact
  task prompt preserving the first user task and the ordered
  user/assistant/tool trajectory. This protocol deviation limits conclusions.
- Assistant spans were constructed by chat-template prefix differencing. The
  Qwen template emitted no assistant mask, so template prefix stability was
  checked and assistant message headers/body/terminators were retained.

## Results

### Reference-based text similarity (100 generated predictions)

| Objective | BERTScore F1 | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
| --- | ---: | ---: | ---: | ---: |
| Assistant-only | **0.5984** | **0.3883** | **0.1955** | **0.2736** |
| Full prompt + assistant | 0.5750 | 0.3523 | 0.1615 | 0.2476 |

These metrics compare the full capped generated trajectory to the full reference
assistant turn, not a final-answer-only span. They are text similarity metrics,
not tool execution success. BERTScore used `microsoft/deberta-xlarge-mnli`,
English, no baseline rescaling. ROUGE uses the `rouge-score` package with Porter
stemming; table reports macro-averaged per-example F1. The exact per-task values
are in `RESULTS/toolbench-large-*-reference-metrics.json`.

Reference-grounded LLM judge (`gpt-4o-mini`, 0–4 scale) was run on the same first
20 tasks for each objective. Assistant-only mean score was **1.40/4** (scores:
0=2, 1=10, 2=6, 3=2); full-loss mean was **1.15/4** (0=5, 1=9, 2=4, 3=2).
This is a small judge sample, not a statistically reliable benchmark result.
The judge compared task-relevant correctness/completeness against the reference
and ignored wording. Full per-task rationales are stored in
`RESULTS/toolbench-large-*-llm-judge.json`. Because reference trajectories may
contain invalid/outdated tool outcomes and generations stop at 64 tokens, judge
scores reflect incomplete responses in this pilot.

| Objective | Train loss | Supervised tokens | Held-out loss | Held-out assistant NLL | Trajectory overlap F1 (n=100) | Final-answer subspan EM | Exact trajectory match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Assistant-only | 0.701 | 242,960 | 0.645 | 0.754 | **0.359** | Not scored: no Finish spans generated | 0 / 100 |
| Full prompt + assistant | 0.806 | 548,071 | 0.701 | **0.747** | 0.330 | Not scored: no Finish spans generated | 0 / 100 |

Both conditions used the same 548,071 input tokens, 512 examples, seed, and 200
updates. Evaluation covers the same 100 held-out tasks. Assistant-only has
0.029 higher mean whole-trajectory token-overlap F1; a paired task bootstrap
95% interval is [0.011, 0.048]. Its mean teacher-forced assistant NLL is 0.0066
higher; paired interval for full minus assistant-only is [-0.0119, -0.0013].
These two signals disagree. Exact trajectory match is 0/100 for both and is
uninformative: it demands character-for-character reproduction of reasoning,
tool syntax, and final answer. Overlap F1 is also only a lexical diagnostic,
not actual task success. The dataset records tool observations but does not
provide a ready-to-run evaluator for these newly generated trajectories. No
ToolBench task-success score is claimed.

Final-answer subspan EM should extract the JSON `final_answer` from `Action:
Finish` in both prediction and reference, then normalize and compare only those
answer strings. Inspection of the saved predictions revealed that neither model
generated a Finish/final_answer span in the first 100 examples. Therefore EM is
**undefined for this generation run**, not 0/28: scoring zero would conflate
failure to reach a final answer (and the 64-token generation cap) with an
incorrect final answer. The emitted `0/28` files from an earlier scorer are
superseded and should not be used. The examples show the models continuing
Thought/Action text at the generation cutoff; they do not provide a final answer
to compare.

Training did optimize: both loss curves declined materially, and held-out loss
is lower than final train loss. That is evidence the code learns the recorded
trajectory distribution, not that it learned to solve tasks. The opposing
held-out NLL and lexical-overlap results, one seed, compacted system prompt,
and lack of execution evaluation prevent a reliable conclusion about which
objective is better. Full loss also changes assistant-token relative weight,
so this is not an isolated prompt-loss effect.

## Artifacts

Adapters and generated predictions are local under ignored `runs/` and
`results/`; raw dataset files are not included. Run metadata and training
history are in each adapter's `run.json`. The local
`data/toolbench-large/manifest.json` records sample counts and split details.
Aggregated metrics are committed as
[`RESULTS/toolbench-large-summary.json`](RESULTS/toolbench-large-summary.json).

## Follow-up required

Pin model and dataset revisions; preserve the upstream tool schema using a
compact, equivalent representation; evaluate parsed action validity and final
answer correctness or run a local deterministic tool simulator; run at least
three seeds; and implement and validate τ-bench and AgentBench adapters
separately before making cross-dataset claims. The current 100-generation
cap was selected because evaluating all 500 examples exceeded a 20-minute
runtime limit.
