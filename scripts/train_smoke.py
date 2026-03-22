from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from limit_metrics import MetricsLogger, summarize_lengths


class TinyTokenDataset(Dataset):
    def __init__(self, records: List[Dict[str, List[int]]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.records[idx]


def collate_fn(batch: List[Dict[str, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_in = max(len(x["input_ids"]) for x in batch)
    max_tgt = max(len(x["target_ids"]) for x in batch)
    inputs = torch.zeros((len(batch), max_in), dtype=torch.long)
    targets = torch.zeros((len(batch), max_tgt), dtype=torch.long)
    for i, ex in enumerate(batch):
        inputs[i, : len(ex["input_ids"])] = torch.tensor(ex["input_ids"], dtype=torch.long)
        targets[i, : len(ex["target_ids"])] = torch.tensor(ex["target_ids"], dtype=torch.long)
    return inputs, targets


class TinyDistillModel(nn.Module):
    def __init__(self, vocab_size: int = 512, hidden: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        pooled = emb.mean(dim=1)
        logits = self.proj(pooled)
        target = target_ids[:, 0].clamp_max(logits.shape[-1] - 1)
        return nn.functional.cross_entropy(logits, target)


def make_synthetic_records(samples: int, seed: int) -> List[Dict[str, List[int]]]:
    random.seed(seed)
    records = []
    for _ in range(samples):
        in_len = random.randint(32, 192)
        tgt_len = random.randint(8, 64)
        records.append(
            {
                "input_ids": [random.randint(1, 511) for _ in range(in_len)],
                "target_ids": [random.randint(1, 511) for _ in range(tgt_len)],
            }
        )
    return records


def dir_stats(path: Path) -> Tuple[int, int]:
    size = 0
    files = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            size += p.stat().st_size
    return size, files


def train(args: argparse.Namespace) -> None:
    logger = MetricsLogger(run_name="tiny_distill_limit")
    logger.memory_snapshot("process_start")

    preprocessing_start = time.perf_counter()
    records = make_synthetic_records(args.samples, args.seed)
    preprocess_s = round(time.perf_counter() - preprocessing_start, 4)
    logger.log("preprocessing_complete", samples=len(records), preprocess_s=preprocess_s)

    input_lens = [len(r["input_ids"]) for r in records]
    target_lens = [len(r["target_ids"]) for r in records]
    seq_stats = {
        "input_tokens": summarize_lengths(input_lens),
        "target_tokens": summarize_lengths(target_lens),
    }
    logger.set("sequence_stats", seq_stats)
    logger.log("sequence_stats", **{f"in_{k}": v for k, v in seq_stats["input_tokens"].items()})
    logger.log("target_stats", **{f"tgt_{k}": v for k, v in seq_stats["target_tokens"].items()})

    data_load_start = time.perf_counter()
    ds = TinyTokenDataset(records)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    data_load_s = round(time.perf_counter() - data_load_start, 4)
    logger.log("data_pipeline_ready", dataloader_init_s=data_load_s)

    model_load_start = time.perf_counter()
    model = TinyDistillModel()
    model_load_s = round(time.perf_counter() - model_load_start, 4)
    logger.log("model_loaded", model_load_s=model_load_s, params=sum(p.numel() for p in model.parameters()))
    logger.memory_snapshot("after_model_load")

    # LoRA wrapping placeholder for smoke test parity; no-op for this tiny model.
    lora_wrap_start = time.perf_counter()
    time.sleep(0.001)
    lora_wrap_s = round(time.perf_counter() - lora_wrap_start, 4)
    logger.log("lora_wrapped", lora_wrap_s=lora_wrap_s)
    logger.memory_snapshot("after_lora_wrap")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    grad_accum = args.grad_accum
    logger.set(
        "training_shape",
        {
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_samples_per_optimizer_step": args.batch_size * grad_accum,
        },
    )

    cumulative_samples = 0
    optimizer_step = 0
    first_fb_captured = False
    step_token_lengths: List[int] = []
    step_clock = time.perf_counter()
    train_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    for i, (input_ids, target_ids) in enumerate(dl, start=1):
        loss = model(input_ids, target_ids) / grad_accum
        loss.backward()

        if not first_fb_captured:
            logger.memory_snapshot("after_first_forward_backward")
            first_fb_captured = True

        if i % grad_accum == 0 or i == len(dl):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            samples_this_step = input_ids.size(0) * (grad_accum if i % grad_accum == 0 else (i % grad_accum))
            cumulative_samples += samples_this_step
            tokens_this_step = int(input_ids.numel() + target_ids.numel())
            step_token_lengths.append(tokens_this_step)
            duration = round(time.perf_counter() - step_clock, 4)
            step_clock = time.perf_counter()

            logger.add_optimizer_step(
                {
                    "optimizer_step": optimizer_step,
                    "step_duration_s": duration,
                    "cumulative_samples": cumulative_samples,
                    "tokens_this_step": tokens_this_step,
                    "loss": round(float(loss.item() * grad_accum), 6),
                }
            )
            logger.memory_snapshot("optimizer_step")

    train_duration = round(time.perf_counter() - train_start, 4)
    logger.set("train_duration_s", train_duration)
    logger.set("tokenization_preprocessing_s", preprocess_s)
    logger.set("dataloader_init_s", data_load_s)
    logger.set("tokens_per_optimizer_step", summarize_lengths(step_token_lengths))
    logger.log("train_complete", train_duration_s=train_duration, optimizer_steps=optimizer_step)

    ckpt_dir = Path(args.output_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.memory_snapshot("before_save")
    save_start = time.perf_counter()
    torch.save(model.state_dict(), ckpt_dir / "adapter.pt")
    (ckpt_dir / "meta.json").write_text(json.dumps({"optimizer_steps": optimizer_step}), encoding="utf-8")
    save_duration = round(time.perf_counter() - save_start, 4)
    size_bytes, file_count = dir_stats(ckpt_dir)
    logger.memory_snapshot("after_save")
    logger.set(
        "checkpoint",
        {
            "save_duration_s": save_duration,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 4),
            "file_count": file_count,
        },
    )
    logger.log("checkpoint_saved", save_duration_s=save_duration, size_mb=round(size_bytes / (1024 * 1024), 4), file_count=file_count)

    logger.memory_snapshot("before_eval")
    reload_start = time.perf_counter()
    reloaded = TinyDistillModel()
    reloaded.load_state_dict(torch.load(ckpt_dir / "adapter.pt", map_location="cpu"))
    reload_s = round(time.perf_counter() - reload_start, 4)

    eval_start = time.perf_counter()
    reloaded.eval()
    with torch.no_grad():
        e_input, e_target = next(iter(dl))
        eval_loss = float(reloaded(e_input, e_target).item())
    eval_s = round(time.perf_counter() - eval_start, 4)

    gen_start = time.perf_counter()
    with torch.no_grad():
        sample_logits = reloaded(e_input[:1], e_target[:1])
        _ = float(sample_logits.item())
    generation_s = round(time.perf_counter() - gen_start, 4)
    logger.memory_snapshot("after_eval")

    logger.set(
        "eval",
        {
            "reload_s": reload_s,
            "eval_duration_s": eval_s,
            "generation_duration_s": generation_s,
            "eval_loss": round(eval_loss, 6),
        },
    )
    logger.log("eval_complete", reload_s=reload_s, eval_s=eval_s, generation_s=generation_s, eval_loss=round(eval_loss, 6))

    logger.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny distillation smoke test with limit-focused logging")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    train(parser.parse_args())
