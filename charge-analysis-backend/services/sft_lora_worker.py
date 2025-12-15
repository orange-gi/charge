"""最小化的 LLaMAFactory-like 训练 worker（sft + lora）。

目标：实现 `llamafactory-cli train deepseek_r1_qwen7b_sft.yaml` 的最小可用能力：
- 读取任务表中的 `hyperparameters.llamafactory_yaml`（YAML 解析后的 dict）
- 读取训练数据集（TrainingDataset.file_path：json 或 jsonl）
- 使用 transformers + peft 做 LoRA SFT 训练（Trainer）
- 按 logging_steps / save_steps 写训练日志与 loss 到数据库（TrainingLog/TrainingMetrics）
- 支持 val_size 切分验证集，并按 eval_steps 记录 eval_loss
- 支持训练完成后用关键词评估接口做离线评估（评估逻辑在 API 中，非 worker）

注意：该项目目录名包含 '-'（charge-analysis-backend），因此 worker 通过 `python -m services.sft_lora_worker`
在该目录作为 cwd 运行，使用相对导入（from database import ...）即可。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from database import session_scope
from models import LogLevel, TrainingLog, TrainingMetrics, TrainingStatus, TrainingTask, TrainingDataset


def _utc_now() -> datetime:
    return datetime.utcnow()


def _append_log(task_id: int, message: str, level: LogLevel = LogLevel.INFO, meta: dict[str, Any] | None = None) -> None:
    with session_scope() as session:
        session.add(
            TrainingLog(
                task_id=task_id,
                log_level=level,
                message=message,
                meta_info=json.dumps(meta) if meta else None,
            )
        )


def _update_task(
    task_id: int,
    *,
    status: TrainingStatus | None = None,
    progress: float | None = None,
    current_epoch: int | None = None,
    current_step: int | None = None,
    total_epochs: int | None = None,
    total_steps: int | None = None,
    metrics: dict[str, Any] | None = None,
    model_path: str | None = None,
    error_message: str | None = None,
    end_time: datetime | None = None,
) -> None:
    with session_scope() as session:
        task = session.get(TrainingTask, task_id)
        if task is None:
            return
        if status is not None:
            task.status = status
        if progress is not None:
            task.progress = float(progress)
        if current_epoch is not None:
            task.current_epoch = int(current_epoch)
        if current_step is not None:
            task.current_step = int(current_step)
        if total_epochs is not None:
            task.total_epochs = int(total_epochs)
        if total_steps is not None:
            task.total_steps = int(total_steps)
        if metrics is not None:
            task.metrics = json.dumps(metrics)
        if model_path is not None:
            task.model_path = model_path
        if error_message is not None:
            task.error_message = error_message
        if end_time is not None:
            task.end_time = end_time
        task.updated_at = _utc_now()
        session.add(task)


def _record_metrics(task_id: int, epoch: int, step: int, payload: dict[str, Any]) -> None:
    with session_scope() as session:
        session.add(
            TrainingMetrics(
                task_id=task_id,
                epoch=epoch,
                step=step,
                loss=payload.get("loss"),
                accuracy=payload.get("accuracy"),
                learning_rate=payload.get("learning_rate"),
                gpu_memory=payload.get("gpu_memory"),
                custom_metrics=json.dumps(payload.get("custom_metrics")) if payload.get("custom_metrics") else None,
            )
        )


def _safe_json_loads(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []

    # JSON array / object
    if raw[0] in "{[":
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []

    # JSONL
    items: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            items.append(obj)
    return items


def _normalize_llamafactory_yaml(cfg: dict[str, Any]) -> dict[str, Any]:
    # 只保留本次需要的最小字段，并提供默认值
    def get(key: str, default: Any = None) -> Any:
        return cfg.get(key, default)

    normalized = {
        # model
        "model_name_or_path": get("model_name_or_path"),
        # method
        "stage": get("stage", "sft"),
        "do_train": bool(get("do_train", True)),
        "finetuning_type": get("finetuning_type", "lora"),
        "lora_target": get("lora_target", "all"),
        "lora_rank": int(get("lora_rank", 8)),
        "lora_alpha": int(get("lora_alpha", 16)),
        "lora_dropout": float(get("lora_dropout", 0.05)),
        # dataset
        "dataset": get("dataset"),
        "template": get("template", "qwen"),
        "cutoff_len": int(get("cutoff_len", 1024)),
        "max_samples": int(get("max_samples", 1000000)),
        "overwrite_cache": bool(get("overwrite_cache", True)),
        "preprocessing_num_workers": int(get("preprocessing_num_workers", 1)),
        # output
        "output_dir": get("output_dir"),
        "logging_steps": int(get("logging_steps", 10)),
        "save_steps": int(get("save_steps", 100)),
        "plot_loss": bool(get("plot_loss", False)),
        "overwrite_output_dir": bool(get("overwrite_output_dir", True)),
        # train
        "per_device_train_batch_size": int(get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(get("gradient_accumulation_steps", 1)),
        "learning_rate": float(get("learning_rate", 5e-4)),
        "num_train_epochs": float(get("num_train_epochs", 1.0)),
        "lr_scheduler_type": get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(get("warmup_ratio", 0.0)),
        "bf16": bool(get("bf16", False)),
        "ddp_timeout": int(get("ddp_timeout", 180000000)),
        # eval
        "val_size": float(get("val_size", 0.0)),
        "per_device_eval_batch_size": int(get("per_device_eval_batch_size", 1)),
        "eval_strategy": get("eval_strategy", "no"),
        "eval_steps": int(get("eval_steps", 50)),
        # misc
        "seed": int(get("seed", 42)),
    }
    if not normalized["model_name_or_path"]:
        raise ValueError("缺少 model_name_or_path")
    if not normalized["output_dir"]:
        raise ValueError("缺少 output_dir")
    if normalized["stage"] != "sft":
        raise ValueError("最小实现仅支持 stage: sft")
    if normalized["finetuning_type"] != "lora":
        raise ValueError("最小实现仅支持 finetuning_type: lora")
    if not normalized["do_train"]:
        raise ValueError("do_train=false 当前不支持")
    return normalized


def _build_qwen_messages(sample: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    system = (sample.get("system") or "").strip()
    instruction = (sample.get("instruction") or "").strip()
    inp = (sample.get("input") or "").strip()
    output = (sample.get("output") or "").strip()

    if instruction and inp:
        user = f"{instruction}\n{inp}"
    else:
        user = instruction or inp

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if user:
        messages.append({"role": "user", "content": user})
    return messages, output


@dataclass
class EncodedSample:
    input_ids: list[int]
    labels: list[int]


class SftJsonDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, samples: list[dict[str, Any]], *, cutoff_len: int, template: str) -> None:
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.template = template
        self.items: list[EncodedSample] = []

        for s in samples:
            encoded = self._encode_one(s)
            if encoded is not None:
                self.items.append(encoded)

    def _encode_one(self, sample: dict[str, Any]) -> EncodedSample | None:
        # 目前最小实现：只实现 template=qwen 的 chat prompt
        messages, answer = _build_qwen_messages(sample)
        if not messages or not answer:
            return None

        tok = self.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        # prompt（到 assistant 起始）
        try:
            prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # fallback：简化格式
            prompt_text = "".join([f"[{m['role'].upper()}]{m['content']}\n" for m in messages]) + "[ASSISTANT]"

        full_text = prompt_text + answer
        if tok.eos_token:
            full_text += tok.eos_token

        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids

        # 截断
        if len(full_ids) > self.cutoff_len:
            full_ids = full_ids[: self.cutoff_len]

        # labels：mask prompt
        labels = [-100] * len(full_ids)
        start = min(len(prompt_ids), len(full_ids))
        if start >= len(full_ids):
            return None
        for i in range(start, len(full_ids)):
            labels[i] = full_ids[i]

        return EncodedSample(input_ids=full_ids, labels=labels)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        return {"input_ids": torch.tensor(item.input_ids, dtype=torch.long), "labels": torch.tensor(item.labels, dtype=torch.long)}


class DataCollatorForCausalLMPadded:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class DBCallback(TrainerCallback):
    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        self._max_steps: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._max_steps = int(state.max_steps) if state.max_steps else None
        _append_log(self.task_id, "训练开始", meta={"max_steps": self._max_steps})
        if self._max_steps:
            _update_task(self.task_id, total_steps=self._max_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        step = int(state.global_step or 0)
        epoch = int(math.floor(state.epoch or 0))

        gpu_mem = None
        if torch.cuda.is_available():
            try:
                gpu_mem = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            except Exception:
                gpu_mem = None

        payload = {
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "gpu_memory": gpu_mem,
            "custom_metrics": {k: v for k, v in logs.items() if k not in {"loss", "learning_rate"}},
        }
        _record_metrics(self.task_id, epoch=epoch, step=step, payload=payload)

        progress = None
        if self._max_steps and self._max_steps > 0:
            progress = min(99.9, (step / self._max_steps) * 100.0)
        _update_task(self.task_id, current_epoch=epoch, current_step=step, progress=progress, metrics=payload)

        if "loss" in logs:
            _append_log(self.task_id, f"step={step} loss={logs.get('loss')}", meta={"logs": logs})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        step = int(state.global_step or 0)
        epoch = int(math.floor(state.epoch or 0))
        payload = {
            "loss": metrics.get("eval_loss"),
            "custom_metrics": metrics,
        }
        _record_metrics(self.task_id, epoch=epoch, step=step, payload=payload)
        _append_log(self.task_id, f"eval@step={step} eval_loss={metrics.get('eval_loss')}", meta=metrics)


def _split_train_eval(items: list[dict[str, Any]], val_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_size <= 0:
        return items, []
    n = len(items)
    k = int(round(n * val_size))
    k = max(1, min(k, n - 1)) if n >= 2 else 0
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    eval_idx = set(idx[:k])
    train, evals = [], []
    for i, it in enumerate(items):
        (evals if i in eval_idx else train).append(it)
    return train, evals


def run(task_id: int) -> None:
    # 读取任务与配置
    with session_scope() as session:
        task = session.get(TrainingTask, task_id)
        if task is None:
            return
        dataset = session.get(TrainingDataset, task.dataset_id) if task.dataset_id else None
        hyper = _safe_json_loads(task.hyperparameters) or {}

    cfg = hyper.get("llamafactory_yaml")
    if not isinstance(cfg, dict):
        _append_log(task_id, "任务缺少 llamafactory_yaml 配置，无法执行真实训练", level=LogLevel.ERROR)
        _update_task(task_id, status=TrainingStatus.FAILED, error_message="missing llamafactory_yaml")
        return

    try:
        cfg = _normalize_llamafactory_yaml(cfg)
    except Exception as exc:
        _append_log(task_id, f"YAML 配置校验失败: {exc}", level=LogLevel.ERROR)
        _update_task(task_id, status=TrainingStatus.FAILED, error_message=str(exc))
        return

    if dataset is None or not dataset.file_path:
        _append_log(task_id, "任务未绑定训练数据集或数据集文件不存在", level=LogLevel.ERROR)
        _update_task(task_id, status=TrainingStatus.FAILED, error_message="missing dataset")
        return

    output_dir = Path(cfg["output_dir"])
    if cfg.get("overwrite_output_dir") and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _append_log(task_id, "开始加载训练数据集", meta={"dataset_path": dataset.file_path})
    all_samples = _read_json_or_jsonl(dataset.file_path)
    if not all_samples:
        _append_log(task_id, "训练数据集为空或解析失败", level=LogLevel.ERROR)
        _update_task(task_id, status=TrainingStatus.FAILED, error_message="empty dataset")
        return

    max_samples = int(cfg.get("max_samples") or len(all_samples))
    all_samples = all_samples[: max_samples]

    train_samples, eval_samples = _split_train_eval(all_samples, float(cfg.get("val_size") or 0.0), int(cfg.get("seed") or 42))
    _append_log(
        task_id,
        "数据集切分完成",
        meta={"total": len(all_samples), "train": len(train_samples), "eval": len(eval_samples)},
    )

    # 模型与 tokenizer
    model_name_or_path = cfg["model_name_or_path"]
    _append_log(task_id, "开始加载 tokenizer / base model（可能较慢）", meta={"model_name_or_path": model_name_or_path})

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    use_bf16 = bool(cfg.get("bf16"))
    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA
    lora_target = cfg.get("lora_target", "all")
    target_modules: Any
    if isinstance(lora_target, str) and lora_target.lower() == "all":
        target_modules = "all-linear"
    elif isinstance(lora_target, str):
        target_modules = [x.strip() for x in lora_target.split(",") if x.strip()]
    else:
        target_modules = lora_target

    lora_config = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # 数据集
    train_ds = SftJsonDataset(tokenizer, train_samples, cutoff_len=int(cfg["cutoff_len"]), template=str(cfg.get("template") or "qwen"))
    eval_ds = (
        SftJsonDataset(tokenizer, eval_samples, cutoff_len=int(cfg["cutoff_len"]), template=str(cfg.get("template") or "qwen"))
        if eval_samples
        else None
    )
    if len(train_ds) == 0:
        _append_log(task_id, "编码后训练样本为 0（可能字段缺失）", level=LogLevel.ERROR)
        _update_task(task_id, status=TrainingStatus.FAILED, error_message="no encoded train samples")
        return

    total_epochs = int(math.ceil(float(cfg["num_train_epochs"])))
    _update_task(task_id, total_epochs=total_epochs, model_path=str(output_dir))

    # TrainingArguments
    eval_strategy = str(cfg.get("eval_strategy") or "no")
    if eval_strategy not in {"no", "steps", "epoch"}:
        eval_strategy = "no"

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        lr_scheduler_type=str(cfg["lr_scheduler_type"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        evaluation_strategy=eval_strategy,
        eval_steps=int(cfg["eval_steps"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        bf16=bool(cfg.get("bf16")) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=(not bool(cfg.get("bf16"))) and torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLMPadded(tokenizer),
        callbacks=[DBCallback(task_id)],
    )

    _append_log(task_id, "开始训练", meta={"output_dir": str(output_dir)})
    start = time.time()
    try:
        trainer.train()
        duration = int(time.time() - start)

        # 保存 LoRA adapter + tokenizer
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # 保存本次训练配置快照（便于复现）
        (output_dir / "llamafactory_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        _update_task(task_id, status=TrainingStatus.COMPLETED, progress=100.0, end_time=_utc_now())
        _append_log(task_id, "训练完成", meta={"duration_seconds": duration, "output_dir": str(output_dir)})
    except Exception as exc:
        _append_log(task_id, f"训练失败: {exc}", level=LogLevel.ERROR)
        _update_task(task_id, status=TrainingStatus.FAILED, error_message=str(exc), end_time=_utc_now())


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args.task_id)


if __name__ == "__main__":
    main()

