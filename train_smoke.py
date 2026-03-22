import argparse
import logging
import math
import random
import resource
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from metrics_logging import MetricsLogger, configure_logging, dir_size_and_files


class ToyDistillDataset(Dataset):
    def __init__(self, n_samples: int, min_len: int, max_len: int, vocab_size: int, seed: int = 7):
        rng = random.Random(seed)
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        for _ in range(n_samples):
            in_len = rng.randint(min_len, max_len)
            tgt_len = rng.randint(min_len, max_len)
            self.inputs.append(torch.tensor([rng.randrange(2, vocab_size) for _ in range(in_len)], dtype=torch.long))
            self.targets.append(torch.tensor([rng.randrange(2, vocab_size) for _ in range(tgt_len)], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


@dataclass
class SeqStats:
    min_len: int
    mean_len: float
    max_len: int
    p95_len: float


class TinyLoRAAdapter(nn.Module):
    def __init__(self, in_dim: int, rank: int = 8):
        super().__init__()
        self.a = nn.Linear(in_dim, rank, bias=False)
        self.b = nn.Linear(rank, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class TinyDistillModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.backbone = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.lora = TinyLoRAAdapter(hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.backbone(x)
        x = x + self.lora(x)
        return self.head(x)


def mem_snapshot(label: str) -> Dict[str, float]:
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"label": label, "max_rss_mb": round(usage_kb / 1024.0, 2)}


def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    ins, tgts = zip(*batch)
    max_in = max(len(x) for x in ins)
    max_t = max(len(x) for x in tgts)

    input_batch = torch.zeros((len(batch), max_in), dtype=torch.long)
    target_batch = torch.zeros((len(batch), max_t), dtype=torch.long)
    input_len = []
    target_len = []

    for i, (inp, tgt) in enumerate(zip(ins, tgts)):
        input_batch[i, : len(inp)] = inp
        target_batch[i, : len(tgt)] = tgt
        input_len.append(len(inp))
        target_len.append(len(tgt))

    return {
        "input_ids": input_batch,
        "target_ids": target_batch,
        "input_lens": torch.tensor(input_len),
        "target_lens": torch.tensor(target_len),
    }


def calc_stats(lengths: List[int]) -> SeqStats:
    ordered = sorted(lengths)
    idx = int(math.ceil(0.95 * len(ordered))) - 1
    return SeqStats(
        min_len=min(ordered),
        mean_len=statistics.fmean(ordered),
        max_len=max(ordered),
        p95_len=ordered[max(0, idx)],
    )


def run(args: argparse.Namespace) -> Path:
    configure_logging()
    metrics = MetricsLogger(run_name="tiny-distill")
    metrics.record("memory", **mem_snapshot("process_start"))

    preprocess_start = time.perf_counter()
    dataset = ToyDistillDataset(args.train_samples, args.min_seq_len, args.max_seq_len, args.vocab_size)
    preprocess_time = time.perf_counter() - preprocess_start
    metrics.record("preprocess", duration_s=round(preprocess_time, 4), samples=len(dataset))

    input_stats = calc_stats([len(x) for x in dataset.inputs])
    target_stats = calc_stats([len(x) for x in dataset.targets])
    metrics.record(
        "sequence_stats",
        input_min=input_stats.min_len,
        input_mean=round(input_stats.mean_len, 2),
        input_max=input_stats.max_len,
        input_p95=input_stats.p95_len,
        target_min=target_stats.min_len,
        target_mean=round(target_stats.mean_len, 2),
        target_max=target_stats.max_len,
        target_p95=target_stats.p95_len,
    )

    model = TinyDistillModel(vocab_size=args.vocab_size, hidden_size=args.hidden_size)
    metrics.record("memory", **mem_snapshot("after_model_load"))
    metrics.record("memory", **mem_snapshot("after_lora_wrapping"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    metrics.record(
        "train_shape",
        per_device_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        effective_samples_per_opt_step=args.batch_size * args.grad_accum,
    )

    run_start = time.perf_counter()
    cumulative_samples = 0
    first_backward_seen = False
    opt_step = 0

    optimizer.zero_grad(set_to_none=True)
    iter_loader = iter(data)
    while True:
        data_load_start = time.perf_counter()
        try:
            batch = next(iter_loader)
        except StopIteration:
            break
        data_load_s = time.perf_counter() - data_load_start
        metrics.record("dataloader", duration_s=round(data_load_s, 5), batch_size=int(batch["input_ids"].shape[0]))

        logits = model(batch["input_ids"])
        target_slice = batch["target_ids"][:, : logits.shape[1]]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_slice.reshape(-1))
        (loss / args.grad_accum).backward()

        if not first_backward_seen:
            metrics.record("memory", **mem_snapshot("after_first_forward_backward"))
            first_backward_seen = True

        cumulative_samples += int(batch["input_ids"].shape[0])

        if cumulative_samples % (args.batch_size * args.grad_accum) == 0:
            step_start = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            step_dur = time.perf_counter() - step_start
            tokens_in_step = int(batch["input_lens"].sum().item() + batch["target_lens"].sum().item())
            metrics.record(
                "optimizer_step",
                optimizer_step=opt_step,
                elapsed_since_run_start_s=round(time.perf_counter() - run_start, 4),
                step_duration_s=round(step_dur, 4),
                cumulative_samples=cumulative_samples,
                tokens_in_optimizer_step=tokens_in_step,
                loss=round(float(loss.item()), 6),
            )
            metrics.record("memory", **mem_snapshot(f"optimizer_step_{opt_step}"))

    metrics.record("memory", **mem_snapshot("before_save"))
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_start = time.perf_counter()
    metrics.record("checkpoint_save_start", checkpoint_dir=str(ckpt_dir))
    torch.save(model.state_dict(), ckpt_dir / "adapter.pt")
    save_duration = time.perf_counter() - save_start
    ckpt_stats = dir_size_and_files(ckpt_dir)
    metrics.record(
        "checkpoint_save_end",
        save_duration_s=round(save_duration, 4),
        checkpoint_bytes=ckpt_stats["bytes"],
        checkpoint_file_count=ckpt_stats["file_count"],
    )
    metrics.record("memory", **mem_snapshot("after_save"))

    metrics.record("memory", **mem_snapshot("before_eval"))
    reload_start = time.perf_counter()
    reloaded = TinyDistillModel(vocab_size=args.vocab_size, hidden_size=args.hidden_size)
    reloaded.load_state_dict(torch.load(ckpt_dir / "adapter.pt", map_location="cpu"))
    reload_dur = time.perf_counter() - reload_start

    eval_start = time.perf_counter()
    reloaded.eval()
    with torch.no_grad():
        eval_batch = next(iter(data))
        eval_logits = reloaded(eval_batch["input_ids"])
        _ = eval_logits.mean().item()
    eval_dur = time.perf_counter() - eval_start

    gen_start = time.perf_counter()
    with torch.no_grad():
        _ = torch.argmax(eval_logits[:, -1, :], dim=-1)
    gen_dur = time.perf_counter() - gen_start

    metrics.record(
        "eval",
        reload_duration_s=round(reload_dur, 4),
        eval_duration_s=round(eval_dur, 4),
        generation_duration_s=round(gen_dur, 4),
    )
    metrics.record("memory", **mem_snapshot("after_eval"))

    metrics.set_summary(
        train_samples=args.train_samples,
        optimizer_steps=opt_step,
        total_runtime_s=round(time.perf_counter() - run_start, 4),
        checkpoint_bytes=ckpt_stats["bytes"],
        input_p95=input_stats.p95_len,
        target_p95=target_stats.p95_len,
    )
    return metrics.write()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny distillation smoke test with limit-focused instrumentation")
    p.add_argument("--train-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--min-seq-len", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=96)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=3200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    artifact = run(parse_args())
    logging.info("completed run; metrics artifact at %s", artifact)
