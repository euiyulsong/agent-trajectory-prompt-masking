# Dataset adapter status

| Benchmark | Status | Notes |
| --- | --- | --- |
| ToolBench | Initial target, adapter pending | API/tool-use trajectories; verify exact release, license, data schema, and environment before training. |
| τ-bench | Planned, not implemented | Stateful domain environments and tool execution; use native simulator and success evaluator, not merely serialized text. |
| AgentBench | Planned, not implemented | Suite of heterogeneous environments; select and pin a specific task/environment before comparing. |

These benchmarks are not a single pooled dataset. Each requires its own split,
normalization, environment pinning, license review, and native evaluation.

## Normalized trajectory schema

The first adapter should emit one JSON object per task:

```json
{
  "task_id": "stable-id",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "name": "search", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {"split": "train", "source": "..."}
}
```

Preserve tool-call structure and control-flow markers when the source format
contains them. Do not flatten structured calls into prose unless the selected
model's chat template requires that representation, and document any conversion.
