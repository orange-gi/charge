"""训练任务模拟服务。"""
from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime

from database import session_scope
from models import TrainingMetrics, TrainingStatus, TrainingTask


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

        try:
            for epoch in range(1, total_epochs + 1):
                for step in range(1, self.steps_per_epoch + 1):
                    await asyncio.sleep(0.1)
                    progress = ((epoch - 1) * self.steps_per_epoch + step) / (
                        total_epochs * self.steps_per_epoch
                    ) * 100
                    metrics = self._generate_metrics(epoch, step)
                    self._update_task(task_id, epoch, step, progress, metrics)

                    if step % 10 == 0:
                        self._record_metrics(task_id, epoch, step, metrics)

            self._mark_completed(task_id)
        except Exception as exc:  # pragma: no cover
            self._mark_failed(task_id, str(exc))

    def _generate_metrics(self, epoch: int, step: int) -> dict:
        loss = max(0.05, 2.5 * (0.9 ** epoch) + random.random() * 0.05)
        accuracy = min(0.99, 0.5 + epoch * 0.05 + random.random() * 0.05)
        return {
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 4),
            "learning_rate": 0.001,
        }

    def _update_task(self, task_id: int, epoch: int, step: int, progress: float, metrics: dict) -> None:
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            task.progress = round(progress, 2)
            task.current_epoch = epoch
            task.current_step = step
            task.metrics = json.dumps(metrics)
            session.add(task)

    def _record_metrics(self, task_id: int, epoch: int, step: int, metrics: dict) -> None:
        with session_scope() as session:
            session.add(
                TrainingMetrics(
                    task_id=task_id,
                    epoch=epoch,
                    step=step,
                    loss=metrics["loss"],
                    accuracy=metrics["accuracy"],
                    learning_rate=metrics["learning_rate"],
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

    def _mark_failed(self, task_id: int, message: str) -> None:
        with session_scope() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            task.status = TrainingStatus.FAILED
            task.error_message = message
            session.add(task)


_training_service: TrainingService | None = None


def get_training_service() -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service
