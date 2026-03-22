from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.metrics_logger import LimitMetricsLogger, directory_size_and_file_count


def get_memory_mb() -> tuple[float, float | None]:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return mem.rss / (1024 * 1024), mem.vms / (1024 * 1024)
    except Exception:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss_kb / 1024.0, None


def summarize_lengths(lengths: list[int]) -> dict[str, float]:
    ordered = sorted(lengths)
    p95_idx = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "min": float(ordered[0]),
        "mean": round(statistics.fmean(ordered), 3),
        "max": float(ordered[-1]),
        "p95": float(ordered[p95_idx]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--metrics-json", type=Path, default=Path("outputs/metrics/run_metrics.json"))
    args = parser.parse_args()

    logger = LimitMetricsLogger(output_path=args.metrics_json)
    logger.log("starting tiny distillation smoke run")
    rss, vms = get_memory_mb()
    logger.add_memory_snapshot("process_start", rss, vms)

    model_name = "google/flan-t5-small"
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.add_event("model_load", duration_s=round(time.perf_counter() - load_started, 6), model_name=model_name)
    rss, vms = get_memory_mb()
    logger.add_memory_snapshot("after_model_load", rss, vms)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    rss, vms = get_memory_mb()
    logger.add_memory_snapshot("after_lora_wrapping", rss, vms)

    synthetic_rows = [
        {"input": f"summarize: sample {i} text about distillation scaling limits", "target": f"sample {i} summary"}
        for i in range(args.samples)
    ]

    preprocess_start = time.perf_counter()
    input_lengths: list[int] = []
    target_lengths: list[int] = []

    def _tokenize(row: dict[str, str]) -> dict[str, list[int]]:
        model_inputs = tokenizer(row["input"], truncation=True, max_length=128)
        labels = tokenizer(text_target=row["target"], truncation=True, max_length=64)
        model_inputs["labels"] = labels["input_ids"]
        input_lengths.append(len(model_inputs["input_ids"]))
        target_lengths.append(len(labels["input_ids"]))
        return model_inputs

    tokenized = Dataset.from_list(synthetic_rows).map(_tokenize)
    preprocess_duration = time.perf_counter() - preprocess_start
    logger.set_data_pipeline(tokenization_s=round(preprocess_duration, 6))
    logger.log(f"tokenization/preprocessing duration={preprocess_duration:.3f}s")

    sequence_input_stats = summarize_lengths(input_lengths)
    sequence_target_stats = summarize_lengths(target_lengths)
    logger.set_sequence_stats(input_stats=sequence_input_stats, target_stats=sequence_target_stats)
    logger.log(
        "sequence stats input(min/mean/max/p95)="
        f"{sequence_input_stats['min']}/{sequence_input_stats['mean']}/{sequence_input_stats['max']}/{sequence_input_stats['p95']} "
        "target(min/mean/max/p95)="
        f"{sequence_target_stats['min']}/{sequence_target_stats['mean']}/{sequence_target_stats['max']}/{sequence_target_stats['p95']}"
    )

    collate_start = time.perf_counter()

    def collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        inputs = tokenizer.pad(
            [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in batch],
            return_tensors="pt",
        )
        label_batch = tokenizer.pad(
            [{"input_ids": row["labels"]} for row in batch],
            return_tensors="pt",
        )
        labels = label_batch["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    dataloader = DataLoader(tokenized, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collate_fn)
    data_load_time = time.perf_counter() - collate_start
    logger.set_data_pipeline(dataloader_build_s=round(data_load_time, 6), num_batches=len(dataloader))
    logger.log(f"dataloader build duration={data_load_time:.3f}s batches={len(dataloader)}")

    effective_samples = args.per_device_batch_size * args.gradient_accumulation_steps
    logger.set_train_shape(
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_samples_per_optimizer_step=effective_samples,
    )
    logger.log(
        f"train shape per_device_batch_size={args.per_device_batch_size} "
        f"grad_accum={args.gradient_accumulation_steps} effective_samples_per_step={effective_samples}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    optimizer_step = 0
    cumulative_samples = 0
    pending_tokens = 0
    first_fb_done = False
    step_start = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader, start=1):
        outputs = model(**batch)
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()

        batch_tokens = int(batch["attention_mask"].sum().item()) + int((batch["labels"] != -100).sum().item())
        pending_tokens += batch_tokens
        cumulative_samples += int(batch["input_ids"].shape[0])

        if not first_fb_done:
            first_fb_done = True
            rss, vms = get_memory_mb()
            logger.add_memory_snapshot("after_first_forward_backward", rss, vms)

        if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(dataloader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            step_duration = time.perf_counter() - step_start
            logger.add_step_metric(
                optimizer_step=optimizer_step,
                step_duration_s=step_duration,
                cumulative_samples=cumulative_samples,
                tokens_this_step=pending_tokens,
            )
            logger.log(
                f"optimizer_step={optimizer_step} step_duration={step_duration:.3f}s "
                f"elapsed={logger.elapsed():.3f}s cumulative_samples={cumulative_samples} "
                f"tokens_this_step={pending_tokens}"
            )
            rss, vms = get_memory_mb()
            logger.add_memory_snapshot(f"optimizer_step_{optimizer_step}", rss, vms)
            pending_tokens = 0
            step_start = time.perf_counter()

    save_dir = args.output_dir / "adapter"
    save_dir.mkdir(parents=True, exist_ok=True)
    rss, vms = get_memory_mb()
    logger.add_memory_snapshot("before_save", rss, vms)

    save_start = time.perf_counter()
    logger.log("checkpoint save started")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_duration = time.perf_counter() - save_start
    ckpt_size, ckpt_files = directory_size_and_file_count(save_dir)
    logger.set_checkpoint_metrics(
        save_started_elapsed_s=round(save_start - logger.run_start, 6),
        save_duration_s=round(save_duration, 6),
        directory_size_bytes=ckpt_size,
        file_count=ckpt_files,
        path=str(save_dir),
    )
    logger.log(
        f"checkpoint save finished duration={save_duration:.3f}s size_bytes={ckpt_size} files={ckpt_files}"
    )
    rss, vms = get_memory_mb()
    logger.add_memory_snapshot("after_save", rss, vms)

    rss, vms = get_memory_mb()
    logger.add_memory_snapshot("before_eval", rss, vms)

    reload_start = time.perf_counter()
    reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    reloaded_model = get_peft_model(reloaded_model, lora_cfg)
    reloaded_model.load_adapter(save_dir, adapter_name="default")
    reload_duration = time.perf_counter() - reload_start

    eval_inputs = tokenizer([row["input"] for row in synthetic_rows[:4]], return_tensors="pt", padding=True, truncation=True)
    eval_start = time.perf_counter()
    with torch.no_grad():
        _ = reloaded_model(**eval_inputs, labels=eval_inputs["input_ids"])
    eval_duration = time.perf_counter() - eval_start

    gen_start = time.perf_counter()
    with torch.no_grad():
        _ = reloaded_model.generate(**eval_inputs, max_new_tokens=12)
    gen_duration = time.perf_counter() - gen_start
    logger.set_eval_metrics(
        reload_s=round(reload_duration, 6),
        eval_forward_s=round(eval_duration, 6),
        generation_s=round(gen_duration, 6),
    )
    logger.log(
        f"eval reload={reload_duration:.3f}s eval_forward={eval_duration:.3f}s generation={gen_duration:.3f}s"
    )

    rss, vms = get_memory_mb()
    logger.add_memory_snapshot("after_eval", rss, vms)

    logger.write()


if __name__ == "__main__":
    main()
