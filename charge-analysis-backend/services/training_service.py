"""训练服务。

历史上该模块仅用于模拟训练（sleep + 随机指标），用于打通前后端流程。
本次改造新增了一个“最小化 LLaMAFactory-like 的 sft lora 训练”实现：
- API 侧：通过 BackgroundTasks 调起本服务（注意 BackgroundTasks 只支持同步 callable）
- 本服务：优先启动独立子进程执行真实训练（transformers + peft），并写入训练日志/指标到数据库
- 兼容：如果任务没有携带训练所需的 YAML/超参配置，则回退到原有模拟流程
"""
from __future__ import annotations

import asyncio
import json
import random
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_settings
from database import session_scope
from models import LogLevel, TrainingLog, TrainingMetrics, TrainingStatus, TrainingTask


class TrainingService:
    def __init__(self, steps_per_epoch: int = 50) -> None:
        self.steps_per_epoch = steps_per_epoch

    def start_training(self, task_id: int) -> None:
        """后台启动训练（供 BackgroundTasks 调用）。

        - 若任务携带了最小训练配置（hyperparameters 内含 llamafactory_yaml），则启动子进程执行真实训练。
        - 否则回退到模拟训练，避免旧数据/旧任务直接报错。
        """
        try:
            hyper = self._get_task_hyperparameters(task_id)
            if hyper and isinstance(hyper, dict) and hyper.get("llamafactory_yaml"):
                self._start_real_training_subprocess(task_id)
                return
        except Exception as exc:  # pragma: no cover
            self._append_log(task_id, f"读取训练配置失败，将回退到模拟训练: {exc}", level=LogLevel.WARNING)

        # fallback：模拟训练（保持历史行为）
        asyncio.run(self._simulate_training(task_id))

    def _get_task_hyperparameters(self, task_id: int) -> dict[str, Any] | None:
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None or not task.hyperparameters:
                return None
            try:
                return json.loads(task.hyperparameters)
            except Exception:
                return None

    def _start_real_training_subprocess(self, task_id: int) -> None:
        settings = get_settings()
        backend_dir = Path(__file__).resolve().parents[1]  # charge-analysis-backend/
        python = sys.executable

        # 预先把任务标为 RUNNING（worker 也会再次校正）
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            task.status = TrainingStatus.RUNNING
            task.start_time = datetime.utcnow()
            task.progress = 0.0
            session.add(task)

        base_upload = Path(settings.upload_path)
        # 统一成绝对路径，避免 cwd 不同导致找不到文件
        if not base_upload.is_absolute():
            base_upload = (Path.cwd() / base_upload).resolve()

        run_dir = base_upload / "training_runs" / str(task_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "worker.log"

        self._append_log(
            task_id,
            "启动真实训练子进程（sft + lora）",
            meta={"cwd": str(backend_dir), "python": python, "worker_log": str(log_path)},
        )

        env = os.environ.copy()

        # 将超时（ddp_timeout）交给 worker 自己处理；这里只负责拉起
        cmd = [python, "-m", "services.sft_lora_worker", "--task-id", str(task_id)]
        try:
            # 先落一行，确保文件必定生成（便于排障：区分“没生成”vs“没写入”）
            run_header = f"[launcher] utc={datetime.utcnow().isoformat()} task_id={task_id}\n".encode("utf-8")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "ab", buffering=0) as fp:
                fp.write(run_header)
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(backend_dir),
                    env=env,
                    stdout=fp,
                    stderr=fp,
                    start_new_session=True,
                )
        except Exception as exc:
            self._append_log(task_id, f"启动训练子进程失败: {exc}", level=LogLevel.ERROR)
            self._mark_failed(task_id, str(exc))
            return

        # 记录运行信息（便于排障）
        try:
            with session_scope() as session:
                task = session.get(TrainingTask, task_id)
                if task is not None:
                    task.logs = json.dumps(
                        {
                            "worker_pid": proc.pid,
                            "worker_cmd": cmd,
                            "worker_cwd": str(backend_dir),
                            "worker_log": str(log_path),
                        },
                        ensure_ascii=False,
                    )
                    session.add(task)
        except Exception:
            pass

        # worker 会自行推进日志/指标与最终状态，这里直接返回即可
        self._append_log(task_id, "训练子进程已启动，开始异步写入训练日志/指标", meta={"pid": proc.pid})

    async def _simulate_training(self, task_id: int) -> None:
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            task.status = TrainingStatus.RUNNING
            task.start_time = datetime.utcnow()
            task.progress = 0.0
            session.add(task)
            total_epochs = task.total_epochs or 10

        self._append_log(task_id, f"任务进入运行状态（模拟训练），计划 {total_epochs} 个 epoch", meta={"total_epochs": total_epochs})

        try:
            total_steps = total_epochs * self.steps_per_epoch
            for epoch in range(1, total_epochs + 1):
                for step in range(1, self.steps_per_epoch + 1):
                    await asyncio.sleep(0.1)
                    global_step = (epoch - 1) * self.steps_per_epoch + step
                    progress = (global_step / total_steps) * 100
                    metrics = self._generate_metrics(epoch, step)
                    self._update_task(task_id, epoch, step, progress, metrics)

                    if step % 5 == 0:
                        self._record_metrics(task_id, epoch, step, metrics)

                self._append_log(
                    task_id,
                    f"Epoch {epoch} 完成，loss={metrics['loss']} accuracy={metrics['accuracy']}",
                    meta={"epoch": epoch, "metrics": metrics},
                )

            self._mark_completed(task_id)
        except Exception as exc:  # pragma: no cover - 保护性分支
            self._append_log(task_id, f"训练失败: {exc}", level=LogLevel.ERROR)
            self._mark_failed(task_id, str(exc))

    def _generate_metrics(self, epoch: int, step: int) -> dict[str, Any]:
        decay = 0.9 ** epoch
        loss = max(0.02, round(2.2 * decay - random.random() * 0.05, 4))
        accuracy = min(0.995, round(0.45 + epoch * 0.05 + random.random() * 0.02, 4))
        gpu_memory = round(6.0 + random.random() * 2.5, 2)  # GB
        return {
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": 0.001,
            "gpu_memory": gpu_memory,
        }

    def _update_task(self, task_id: int, epoch: int, step: int, progress: float, metrics: dict[str, Any]) -> None:
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            task.progress = round(progress, 2)
            task.current_epoch = epoch
            task.current_step = step
            task.metrics = json.dumps(metrics)
            session.add(task)

    def _record_metrics(self, task_id: int, epoch: int, step: int, metrics: dict[str, Any]) -> None:
        with session_scope() as session:
            session.add(
                TrainingMetrics(
                    task_id=task_id,
                    epoch=epoch,
                    step=step,
                    loss=metrics.get("loss"),
                    accuracy=metrics.get("accuracy"),
                    learning_rate=metrics.get("learning_rate"),
                    gpu_memory=metrics.get("gpu_memory"),
                )
            )

    def _append_log(
        self,
        task_id: int,
        message: str,
        level: LogLevel = LogLevel.INFO,
        meta: dict[str, Any] | None = None,
    ) -> None:
        with session_scope() as session:
            session.add(
                TrainingLog(
                    task_id=task_id,
                    log_level=level,
                    message=message,
                    meta_info=json.dumps(meta) if meta else None,
                )
            )

    def _mark_completed(self, task_id: int) -> None:
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            task.status = TrainingStatus.COMPLETED
            task.progress = 100.0
            task.end_time = datetime.utcnow()
            task.model_path = task.model_path or f"/models/task_{task_id}.bin"
            session.add(task)
        self._append_log(task_id, "训练完成，模型已生成", level=LogLevel.INFO)

    def _mark_failed(self, task_id: int, message: str) -> None:
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            task.status = TrainingStatus.FAILED
            task.error_message = message
            session.add(task)

    def log_event(self, task_id: int, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """供外部调用记录事件。"""
        self._append_log(task_id, message, level=level)


_training_service: TrainingService | None = None


def get_training_service() -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service
