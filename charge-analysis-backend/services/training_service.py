"""训练任务模拟服务，负责异步推进任务、记录指标与日志。"""
from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime
from typing import Any

from database import session_scope
from models import LogLevel, TrainingLog, TrainingMetrics, TrainingStatus, TrainingTask


class TrainingService:
    def __init__(self, steps_per_epoch: int = 50) -> None:
        self.steps_per_epoch = steps_per_epoch

    async def start_training(self, task_id: int) -> None:
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            task.status = TrainingStatus.RUNNING
            task.start_time = datetime.utcnow()
            task.progress = 0.0
            session.add(task)
            total_epochs = task.total_epochs or 10
        self._append_log(
            task_id,
            f"任务进入运行状态，计划 {total_epochs} 个 epoch",
            meta={"total_epochs": total_epochs},
        )

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
