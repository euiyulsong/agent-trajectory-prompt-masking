# Experiment protocol

## Question

Does supervising prompt/tool-observation tokens change agent performance when
fine-tuning on the same agentic trajectories?

## Primary pilot matrix

| Factor | Values |
| --- | --- |
| Dataset | One audited adapter per run (initial target: ToolBench) |
| Format | Full recorded trajectory, same for both objectives |
| Objective | `assistant_only`, `full` |
| Seeds | At least 3 for a directional claim; 1 only for pipeline smoke |
| Model | Same pinned causal LM revision for all conditions |
| LoRA | Same config and target modules for all conditions |

Keep train/eval task IDs disjoint and evaluation fixed. Match optimizer update
counts across conditions. Do not silently truncate examples differently by
condition: filter using the maximum rendered length across conditions before
training and report the excluded fraction. Use the benchmark's own task success
evaluation where possible; compare outcomes paired by task and seed.

## First ToolBench pilot run

The initial 57-train / 27-held-out pilot, its exact deviations, and descriptive
results are documented in [`../REPORT.md`](../REPORT.md). In particular, it is
not an official ToolBench task-success evaluation: the oversized static system
tool schema was compacted, and the separate official test split was not run.

## Objectives

- `assistant_only`: label assistant turns only, including assistant headers and
  end markers. User and tool messages receive `-100` labels.
- `full`: label all non-padding tokens.

With ordinary token-mean cross entropy, full loss also changes assistant-token
weight. State this limitation. A separate normalized loss is needed to isolate
the prompt-loss contribution.

## Outcomes

Primary: benchmark-native task success / reward. Secondary: tool-call parse or
schema validity, completion rate, assistant-token loss, prompt/tool-token loss,
input and supervised tokens, wall time, and failures by trajectory turn.

Use paired task bootstrap intervals within each seed, then show each seed's
effect. Do not treat examples from one task as independent when the benchmark
groups them under a shared scenario. Clearly label intervals as task-sampling
uncertainty; they do not include training-seed uncertainty.

## Data hygiene and safety

Preserve upstream attribution and license notices. Do not commit raw datasets,
credentials, private traces, or generated outputs containing sensitive data.
Review benchmark licenses and tool side effects before execution. Use sandboxed
or mock tools for smoke tests; never run untrusted tool trajectories against
real accounts.

## Result claims

Separate completed runs from planned protocol. A smoke test validates software,
not model behavior. Report hardware, exact revisions, seed, sample sizes,
excluded examples, and any deviations. Avoid generalizing one benchmark to
agentic tasks broadly.
