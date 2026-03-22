import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src_metrics_logger import MetricsLogger


class TinyTokenizer:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_len: int) -> list[int]:
        toks = [((hash(t) % (self.vocab_size - 2)) + 2) for t in text.split()]
        toks = toks[:max_len]
        if not toks:
            toks = [1]
        return toks


class PromptDataset(Dataset):
    def __init__(self, enc_pairs: list[tuple[list[int], list[int]]]):
        self.enc_pairs = enc_pairs

    def __len__(self):
        return len(self.enc_pairs)

    def __getitem__(self, idx: int):
        return self.enc_pairs[idx]


@dataclass
class LoRAConfig:
    r: int = 4
    alpha: float = 8.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(base.in_features, cfg.r) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(cfg.r, base.out_features))
        self.scale = cfg.alpha / cfg.r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a @ self.lora_b) * self.scale


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.ff = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        x = torch.tanh(self.ff(x))
        logits = self.head(x)
        return logits



def collate(batch):
    max_in = max(len(x[0]) for x in batch)
    max_tg = max(len(x[1]) for x in batch)
    max_len = max(max_in, max_tg)
    inputs, targets, in_lens, tg_lens = [], [], [], []
    for inp, tgt in batch:
        in_lens.append(len(inp))
        tg_lens.append(len(tgt))
        inp_pad = inp + [0] * (max_len - len(inp))
        tgt_pad = tgt + [0] * (max_len - len(tgt))
        inputs.append(inp_pad)
        targets.append(tgt_pad)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "target_ids": torch.tensor(targets, dtype=torch.long),
        "input_lens": in_lens,
        "target_lens": tg_lens,
    }


def make_pairs(n: int) -> list[tuple[str, str]]:
    pairs = []
    for i in range(n):
        base = f"summarize sample {i} with compact rationale and decision"
        extra = " ".join(["token"] * random.randint(4, 48))
        prompt = f"{base} {extra}"
        target = f"result {i} {' '.join(['answer'] * random.randint(3, 30))}"
        pairs.append((prompt, target))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    mlog = MetricsLogger(output_dir=args.outputs / "metrics")
    mlog.snapshot_memory("process_start")

    tokenizer = TinyTokenizer()
    raw_pairs = make_pairs(args.samples)
    split = int(0.85 * len(raw_pairs))
    train_raw, eval_raw = raw_pairs[:split], raw_pairs[split:]

    t0 = time.perf_counter()
    train_encoded = [
        (tokenizer.encode(p, args.max_len), tokenizer.encode(t, args.max_len)) for p, t in train_raw
    ]
    eval_encoded = [
        (tokenizer.encode(p, args.max_len), tokenizer.encode(t, args.max_len)) for p, t in eval_raw
    ]
    preprocess_s = time.perf_counter() - t0
    mlog.set_metric("preprocessing_time_s", round(preprocess_s, 4))
    mlog.log_event("preprocessing_complete", seconds=round(preprocess_s, 4), samples=args.samples)

    train_in_lens = [len(x[0]) for x in train_encoded]
    train_tg_lens = [len(x[1]) for x in train_encoded]
    mlog.set_metric("input_len_stats", mlog.sequence_stats(train_in_lens))
    mlog.set_metric("target_len_stats", mlog.sequence_stats(train_tg_lens))
    mlog.log_event("sequence_stats", input_stats=mlog.sequence_stats(train_in_lens), target_stats=mlog.sequence_stats(train_tg_lens))

    train_ds = PromptDataset(train_encoded)
    eval_ds = PromptDataset(eval_encoded)

    t0 = time.perf_counter()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    dataloader_build_s = time.perf_counter() - t0
    mlog.set_metric("dataloader_build_time_s", round(dataloader_build_s, 4))

    t0 = time.perf_counter()
    _ = next(iter(train_dl))
    first_batch_load_s = time.perf_counter() - t0
    mlog.set_metric("first_batch_load_time_s", round(first_batch_load_s, 4))
    mlog.log_event("dataloader_ready", build_s=round(dataloader_build_s, 4), first_batch_load_s=round(first_batch_load_s, 4))

    model = TinyLM(vocab_size=512)
    teacher = TinyLM(vocab_size=512)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    mlog.snapshot_memory("after_model_load")

    model.head = LoRALinear(model.head, LoRAConfig())
    mlog.snapshot_memory("after_lora_wrap")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-3)

    per_device_batch = args.batch_size
    grad_accum = args.grad_accum
    eff_samples_per_step = per_device_batch * grad_accum
    mlog.set_metric("training_shape", {
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_samples_per_optimizer_step": eff_samples_per_step,
    })
    mlog.log_event(
        "training_shape",
        per_device_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        effective_samples_per_optimizer_step=eff_samples_per_step,
    )

    global_step = 0
    opt_step = 0
    samples_processed = 0
    run_train_start = time.perf_counter()
    first_fb_done = False
    step_times = []
    tokens_per_step = []
    train_loss_sum = 0.0

    for _ in range(args.epochs):
        optim.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        step_token_count = 0
        for batch_idx, batch in enumerate(train_dl, start=1):
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            with torch.no_grad():
                t_logits = teacher(input_ids)

            s_logits = model(input_ids)
            kl = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            )
            ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), target_ids.view(-1), ignore_index=0)
            loss = (0.7 * kl + 0.3 * ce) / grad_accum
            loss.backward()
            train_loss_sum += loss.item() * grad_accum

            if not first_fb_done:
                mlog.snapshot_memory("after_first_forward_backward")
                first_fb_done = True

            samples_processed += input_ids.size(0)
            global_step += 1
            step_token_count += int(sum(batch["input_lens"]) + sum(batch["target_lens"]))

            if global_step % grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1
                step_duration = time.perf_counter() - step_start
                step_times.append(step_duration)
                tokens_per_step.append(step_token_count)
                mlog.snapshot_memory("optimizer_step")
                mlog.log_event(
                    "optimizer_step",
                    optimizer_step=opt_step,
                    cumulative_samples=samples_processed,
                    step_duration_s=round(step_duration, 4),
                    tokens_this_step=step_token_count,
                )
                step_start = time.perf_counter()
                step_token_count = 0

    train_duration = time.perf_counter() - run_train_start
    mlog.set_metric("train_duration_s", round(train_duration, 4))
    mlog.set_metric("optimizer_step_time_s", {
        "mean": round(sum(step_times) / max(1, len(step_times)), 4),
        "max": round(max(step_times) if step_times else 0.0, 4),
        "count": len(step_times),
    })
    mlog.set_metric("tokens_per_optimizer_step", mlog.sequence_stats(tokens_per_step))
    mlog.set_metric("train_loss_avg", round(train_loss_sum / max(1, global_step), 6))
    mlog.log_event("train_complete", duration_s=round(train_duration, 4), optimizer_steps=opt_step)

    ckpt_dir = args.outputs / "adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mlog.snapshot_memory("before_save")
    save_start = time.perf_counter()
    mlog.log_event("checkpoint_save_start", path=str(ckpt_dir))
    torch.save({"head_lora_a": model.head.lora_a.detach(), "head_lora_b": model.head.lora_b.detach()}, ckpt_dir / "adapter.pt")
    save_duration = time.perf_counter() - save_start
    size_bytes, file_count = mlog.directory_size(ckpt_dir)
    mlog.snapshot_memory("after_save")
    mlog.set_metric("checkpoint", {
        "save_duration_s": round(save_duration, 4),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 4),
        "file_count": file_count,
    })
    mlog.log_event(
        "checkpoint_save_end",
        save_duration_s=round(save_duration, 4),
        checkpoint_size_mb=round(size_bytes / (1024 * 1024), 4),
        file_count=file_count,
    )

    mlog.snapshot_memory("before_eval")
    reload_start = time.perf_counter()
    reloaded = TinyLM(vocab_size=512)
    reloaded.head = LoRALinear(reloaded.head, LoRAConfig())
    adapter_state = torch.load(ckpt_dir / "adapter.pt", map_location="cpu")
    with torch.no_grad():
        reloaded.head.lora_a.copy_(adapter_state["head_lora_a"])
        reloaded.head.lora_b.copy_(adapter_state["head_lora_b"])
    reload_duration = time.perf_counter() - reload_start

    eval_start = time.perf_counter()
    reloaded.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_dl:
            logits = reloaded(batch["input_ids"])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["target_ids"].view(-1),
                ignore_index=0,
            )
            losses.append(loss.item())
    eval_duration = time.perf_counter() - eval_start

    gen_start = time.perf_counter()
    with torch.no_grad():
        sample = next(iter(eval_dl))["input_ids"][0:1]
        _ = reloaded(sample).argmax(dim=-1)
    gen_duration = time.perf_counter() - gen_start

    mlog.snapshot_memory("after_eval")
    mlog.set_metric("eval", {
        "reload_duration_s": round(reload_duration, 4),
        "eval_duration_s": round(eval_duration, 4),
        "generation_duration_s": round(gen_duration, 4),
        "eval_loss": round(sum(losses) / max(1, len(losses)), 6),
    })
    mlog.log_event(
        "eval_complete",
        reload_duration_s=round(reload_duration, 4),
        eval_duration_s=round(eval_duration, 4),
        generation_duration_s=round(gen_duration, 4),
    )

    mlog.write_summary()


if __name__ == "__main__":
    main()
