#!/usr/bin/env python3
"""Masking primitives and offline smoke tests for agent trajectory SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def encode_messages(tokenizer: Any, messages: list[dict[str, Any]], objective: str) -> dict[str, list[int]]:
    """Encode a full chat and construct labels with validated message spans.

    Assistant-only masking uses the tokenizer's native assistant-token mask if
    supported. Otherwise it fails closed rather than guessing token boundaries.
    """
    if objective not in {"assistant_only", "full"}:
        raise ValueError(f"Unknown objective: {objective}")
    if objective == "assistant_only" and not getattr(tokenizer, "supports_assistant_mask", False):
        raise ValueError(
            "Tokenizer/template must expose validated assistant token spans; "
            "refusing approximate boundary masking"
        )

    if objective == "assistant_only":
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = list(encoded["input_ids"])
        mask = list(encoded.get("assistant_masks", encoded.get("assistant_tokens_mask", [])))
        if len(mask) != len(ids) or not any(mask):
            raise ValueError("Missing or invalid assistant token mask")
        labels = [token if supervised else -100 for token, supervised in zip(ids, mask)]
    else:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        ids = list(encoded["input_ids"])
        labels = ids.copy()

    if not ids or len(ids) != len(labels):
        raise ValueError("Invalid input/label lengths")
    if not any(label != -100 for label in labels):
        raise ValueError("No supervised tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class FakeTokenizer:
    """Deterministic tokenizer stub used only to test masking mechanics."""

    supports_assistant_mask = True

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
    ) -> Any:
        del tokenize, add_generation_prompt
        ids: list[int] = []
        mask: list[int] = []
        for msg in messages:
            role = str(msg["role"])
            # Message markers, role, content and end marker are independently
            # tokenized here to make expected supervision unambiguous.
            part = [1, 10 + len(role), 2, *[100 + ord(c) for c in msg["content"]], 3]
            ids.extend(part)
            mask.extend([int(role == "assistant")] * len(part))
        if return_assistant_tokens_mask:
            return {"input_ids": ids, "assistant_masks": mask}
        return {"input_ids": ids} if return_dict else ids


def smoke() -> None:
    messages = [
        {"role": "user", "content": "Find the answer."},
        {"role": "assistant", "content": "call search"},
        {"role": "tool", "name": "search", "content": "Found it."},
        {"role": "assistant", "content": "The answer is 42."},
    ]
    tokenizer = FakeTokenizer()
    assistant = encode_messages(tokenizer, messages, "assistant_only")
    full = encode_messages(tokenizer, messages, "full")
    assert len(assistant["input_ids"]) == len(assistant["labels"])
    assert len(full["input_ids"]) == len(full["labels"])
    assert sum(label != -100 for label in assistant["labels"]) < len(full["labels"])
    assert any(label != -100 for label in assistant["labels"])
    print(json.dumps({
        "status": "ok",
        "tokens": len(full["input_ids"]),
        "assistant_loss_tokens": sum(label != -100 for label in assistant["labels"]),
        "full_loss_tokens": sum(label != -100 for label in full["labels"]),
        "note": "Synthetic smoke test only; no benchmark/model result",
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "validate-jsonl"])
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    else:
        if args.input is None:
            parser.error("validate-jsonl requires --input")
        count = 0
        with args.input.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if not isinstance(row.get("task_id"), str) or not row["task_id"]:
                    raise ValueError(f"Line {line_number}: missing task_id")
                if not isinstance(row.get("messages"), list) or not row["messages"]:
                    raise ValueError(f"Line {line_number}: missing messages")
                if any(not isinstance(m.get("role"), str) or "content" not in m for m in row["messages"]):
                    raise ValueError(f"Line {line_number}: malformed message")
                count += 1
        print(f"Validated {count} tasks")


if __name__ == "__main__":
    main()
