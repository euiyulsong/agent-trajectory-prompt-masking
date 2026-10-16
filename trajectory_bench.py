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
    if objective == "assistant_only":
        if getattr(tokenizer, "supports_assistant_mask", False):
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
        else:
            ids, mask = assistant_mask_by_prefix(tokenizer, messages)
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


def assistant_mask_by_prefix(tokenizer: Any, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Find assistant spans by prefix differencing and fail on unstable templates.

    This follows the supported Qwen-style invariant: appending a new message may
    only append tokens; it cannot rewrite any previously rendered prefix.
    Assistant headers, body, and message terminators are supervised.
    """
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(full, dict):
        full = full["input_ids"]
    ids = list(full)
    mask = [0] * len(ids)
    starts: list[int] = []
    previous: list[int] = []
    for index, message in enumerate(messages):
        prefix = tokenizer.apply_chat_template(
            messages[: index + 1], tokenize=True, add_generation_prompt=False
        )
        if isinstance(prefix, dict):
            prefix = prefix["input_ids"]
        prefix = list(prefix)
        if prefix[: len(previous)] != previous or ids[: len(prefix)] != prefix:
            raise ValueError("Chat template rewrites prior tokens; cannot safely mask by prefix")
        starts.append(len(previous))
        if index > 0:
            if messages[index - 1]["role"] == "assistant":
                mask[starts[-2] : starts[-1]] = [1] * (starts[-1] - starts[-2])
        previous = prefix
    if messages[-1]["role"] == "assistant":
        mask[starts[-1] :] = [1] * (len(ids) - starts[-1])
    if len(previous) != len(ids):
        # Some templates add a final assistant terminator only after all messages.
        last = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if isinstance(last, dict):
            last = last["input_ids"]
        if list(last) != ids:
            raise ValueError("Inconsistent template rendering while constructing assistant spans")
        if messages[-1]["role"] == "assistant":
            mask[len(previous):] = [1] * (len(ids) - len(previous))
    return ids, mask


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


def load_toolbench_json(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load ToolBench's conversations/from/value JSON into chat messages."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw[:limit] if limit else raw
    normalized = []
    role_map = {"function": "tool", "tool": "tool", "assistant": "assistant", "user": "user", "system": "system"}
    for row in rows:
        messages = []
        for turn in row["conversations"]:
            role = role_map.get(turn["from"])
            if role is None:
                raise ValueError(f"Unknown ToolBench role: {turn['from']}")
            message = {"role": role, "content": turn["value"]}
            if role == "tool":
                message["name"] = turn.get("name", "tool")
            messages.append(message)
        normalized.append({"task_id": str(row["id"]), "messages": messages})
    return normalized


def prepare_toolbench(args: argparse.Namespace) -> None:
    """Make a deterministic, task-disjoint small pilot from downloaded JSON."""
    from transformers import AutoTokenizer

    train_all = load_toolbench_json(args.train_json)
    official_test = load_toolbench_json(args.test_json)
    test_ids = {row["task_id"] for row in official_test}
    import random

    candidates = [row for row in train_all if row["task_id"] not in test_ids]
    random.Random(args.seed).shuffle(candidates)
    train = candidates[: args.train_examples]
    eval_rows = candidates[args.train_examples : args.train_examples + args.eval_examples]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_length = args.max_length
    kept_train, kept_eval = [], []
    for row, dest in [(r, kept_train) for r in train] + [(r, kept_eval) for r in eval_rows]:
        # Filter long serialized tool schemas and trajectories. Supervision and
        # all user/tool context remain intact for each retained task.
        messages = row["messages"]
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        rendered = [
            ({"role": "system", "content": "You are an agent. Solve the user's task using the available tool trajectory.\nTask: " + first_user}
             if msg["role"] == "system" else msg)
            for msg in messages
        ]
        ids = tokenizer.apply_chat_template(rendered, tokenize=True, add_generation_prompt=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if len(ids) <= max_length and row["messages"][-1]["role"] == "assistant":
            dest.append(row)
    # The source includes very large function documentation in the system
    # message. For the pilot only, retain the task instructions from the
    # original user message and keep the trajectory observations/calls intact.
    # This avoids wasting the context budget on repeated static tool schemas.
    def compact_system(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        compact = []
        for msg in messages:
            if msg["role"] == "system":
                content = "You are an agent. Solve the user's task using the available tool trajectory.\nTask: " + first_user
                compact.append({"role": "system", "content": content})
            else:
                compact.append(msg)
        return {**row, "messages": compact}

    kept_train = [compact_system(row) for row in kept_train]
    kept_eval = [compact_system(row) for row in kept_eval]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", kept_train), ("eval", kept_eval)]:
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "ToolBench",
        "source": "Yhyu13/ToolBench_toolllama_G123_dfs",
        "model": args.model,
        "seed": args.seed,
        "requested_train": args.train_examples,
        "requested_eval": args.eval_examples,
        "train_count": len(kept_train),
        "eval_count": len(kept_eval),
        "max_length": max_length,
        "excluded_for_length": args.train_examples + args.eval_examples - len(kept_train) - len(kept_eval),
        "official_test_count": len(official_test),
        "split_note": "Fixed official test excluded from this train/eval pilot; task IDs kept disjoint.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(args.data_dir / "train.jsonl")
    encoded = [encode_messages(tokenizer, row["messages"], args.objective) for row in rows]
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"
    ))
    model.enable_input_require_grads()
    model.config.use_cache = False

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in batch)
        result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in batch:
            pad = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [tokenizer.pad_token_id] * pad)
            result["attention_mask"].append(row["attention_mask"] + [0] * pad)
            result["labels"].append(row["labels"] + [-100] * pad)
        return {key: torch.tensor(value) for key, value in result.items()}

        training_args = TrainingArguments(
        output_dir=str(args.output), per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=max(1, args.max_steps // 20),
        logging_steps=5, save_strategy="no", report_to="none", bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True, remove_unused_columns=False, seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset.from_list(encoded), data_collator=collate)
    train_result = trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    run = {
        "objective": args.objective, "seed": args.seed, "model": args.model,
        "train_examples": len(rows), "max_steps": args.max_steps,
        "input_tokens": sum(len(row["input_ids"]) for row in encoded),
        "loss_tokens": sum(sum(token != -100 for token in row["labels"]) for row in encoded),
        "train_loss": train_result.metrics.get("train_loss"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2))
    print(json.dumps(run, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import math
    import re
    import string

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    rows = read_jsonl(args.data_dir / "eval.jsonl")
    outputs = []
    for row in rows:
        prompt = row["messages"][:-1]
        encoded = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        encoded = encoded.to(device)
        with torch.no_grad():
            generated = model.generate(
                encoded, attention_mask=torch.ones_like(encoded), max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(generated[0, encoded.shape[1]:], skip_special_tokens=True)
        target = row["messages"][-1]["content"] if row["messages"][-1]["role"] == "assistant" else ""
        target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        prompt_for_loss = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(prompt_for_loss, dict):
            prompt_for_loss = prompt_for_loss["input_ids"]
        joint = torch.cat([prompt_for_loss.to(device), target_ids], dim=1)
        labels = torch.full_like(joint, -100)
        labels[:, prompt_for_loss.shape[1]:] = target_ids
        with torch.no_grad():
            out = model(input_ids=joint, attention_mask=torch.ones_like(joint), labels=labels)
        # Strict string equality is not meaningful for ToolBench trajectories:
        # they include long natural-language reasoning, tool syntax, and final
        # answers. Report a normalized token-overlap score as a diagnostic,
        # alongside teacher-forced token NLL; not as benchmark task success.
        def normalize(text: str) -> list[str]:
            text = text.lower().translate(str.maketrans("", "", string.punctuation))
            return re.findall(r"\w+", text)

        pred_tokens, target_tokens = normalize(pred), normalize(target)
        common = sum((__import__("collections").Counter(pred_tokens) & __import__("collections").Counter(target_tokens)).values())
        overlap_f1 = 2 * common / (len(pred_tokens) + len(target_tokens)) if pred_tokens or target_tokens else 1.0
        outputs.append({
            "task_id": row["task_id"], "prediction": pred, "target": target,
            "target_token_count": int(target_ids.shape[1]),
            "assistant_nll": float(out.loss),
            "exact_match": float(pred.strip() == target.strip()),
            "token_overlap_f1": overlap_f1,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in outputs) + "\n")
    print(f"Wrote {len(outputs)} predictions to {args.output}")


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
    parser.add_argument("command", choices=["smoke", "validate-jsonl", "prepare-toolbench", "train", "evaluate"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--train-json", type=Path)
    parser.add_argument("--test-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--objective", choices=["assistant_only", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-examples", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "validate-jsonl":
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
    elif args.command == "prepare-toolbench":
        prepare_toolbench(args)
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
