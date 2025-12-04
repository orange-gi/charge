"""充电分析调度与结果持久化服务。"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from database import session_scope
from langgraph_workflow import ChargingAnalysisWorkflow, AnalysisStatus as WorkflowStatus
from models import AnalysisResult, AnalysisStatus, ChargingAnalysis

logger = logging.getLogger(__name__)
class AnalysisService:
    """负责协调 LangGraph 工作流与数据库交互。"""

    def __init__(self) -> None:
        self.workflow = ChargingAnalysisWorkflow()

    async def run_analysis(self, analysis_id: int) -> None:
        """执行工作流并将结果写入数据库。"""
        logger.info("开始执行分析 %s", analysis_id)
        initial_state = self._build_initial_state(analysis_id)
        if initial_state is None:
            logger.error("分析 %s 不存在，终止", analysis_id)
            return

        try:
            final_state = await self.workflow.execute(initial_state)
            await self._persist_success_state(analysis_id, final_state)
            logger.info("分析 %s 完成", analysis_id)
        except Exception as exc:  # pragma: no cover - 仅用于兜底
            logger.exception("分析 %s 执行失败: %s", analysis_id, exc)
            self._mark_failed(analysis_id, str(exc))

    def _build_initial_state(self, analysis_id: int) -> Dict[str, Any] | None:
        with session_scope() as session:
            analysis: ChargingAnalysis | None = session.get(ChargingAnalysis, analysis_id)
            if analysis is None:
                return None

            analysis.status = AnalysisStatus.PROCESSING
            analysis.progress = 5.0
            analysis.started_at = analysis.started_at or datetime.utcnow()

            session.add(analysis)

        file_path = Path(analysis.file_path)
        progress_callback = self._progress_callback_factory(analysis_id)

        return {
            "analysis_id": str(analysis_id),
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": analysis.file_size or 0,
            "user_id": analysis.user_id,
            "validation_status": "pending",
            "parsing_status": "pending",
            "iteration": 0,
            "analysis_status": WorkflowStatus.PENDING,
            "progress_callback": progress_callback,
            "start_time": datetime.utcnow(),
        }

    def _progress_callback_factory(self, analysis_id: int):
        async def _progress(stage: str, progress: float, message: str | None = None) -> None:
            logger.info("分析 %s 进度 %s%% - %s", analysis_id, progress, stage)
            with session_scope() as session:
                analysis = session.get(ChargingAnalysis, analysis_id)
                if analysis is None:
                    return
                analysis.status = AnalysisStatus.PROCESSING
                analysis.progress = float(progress)
                analysis.updated_at = datetime.utcnow()
                session.add(analysis)
        return _progress

    async def _persist_success_state(self, analysis_id: int, state: Dict[str, Any]) -> None:
        summary_payload = self._build_result_payload(state)
        llm_analysis = state.get("llm_analysis") or {}
        recommendations = llm_analysis.get("recommendations") or []
        findings = llm_analysis.get("findings") or []

        with session_scope() as session:
            analysis: ChargingAnalysis | None = session.get(ChargingAnalysis, analysis_id)
            if analysis is None:
                return

            workflow_status: WorkflowStatus = state.get("analysis_status", WorkflowStatus.COMPLETED)
            analysis.status = self._map_workflow_status(workflow_status)
            analysis.progress = 100.0 if analysis.status == AnalysisStatus.COMPLETED else analysis.progress
            analysis.completed_at = datetime.utcnow()
            analysis.result_data = json.dumps(summary_payload, ensure_ascii=False)

            # 清理旧结果
            session.query(AnalysisResult).filter(
                AnalysisResult.analysis_id == analysis_id
            ).delete(synchronize_session=False)

            # 汇总结果
            session.add(
                AnalysisResult(
                    analysis_id=analysis_id,
                    result_type="summary",
                    title="充电分析总结",
                    content=llm_analysis.get("summary") or summary_payload.get("final_report", {}).get("summary", ""),
                    confidence_score=summary_payload.get("flow_analysis", {}).get("confidence", 0.0),
                    metadata=json.dumps({"category": "overview", "risk": llm_analysis.get("risk_assessment")}, ensure_ascii=False),
                )
            )

            for idx, finding in enumerate(findings, start=1):
                session.add(
                    AnalysisResult(
                        analysis_id=analysis_id,
                        result_type="finding",
                        title=f"关键发现{idx}",
                        content=finding,
                        confidence_score=0.85,
                        metadata=json.dumps({"category": "finding"}, ensure_ascii=False),
                    )
                )

            for idx, rec in enumerate(recommendations, start=1):
                session.add(
                    AnalysisResult(
                        analysis_id=analysis_id,
                        result_type="recommendation",
                        title=f"建议措施{idx}",
                        content=rec,
                        confidence_score=0.8,
                        metadata=json.dumps({"priority": "medium"}, ensure_ascii=False),
                    )
                )

            data_stats = summary_payload.get("data_stats")
            if data_stats:
                session.add(
                    AnalysisResult(
                        analysis_id=analysis_id,
                        result_type="technical",
                        title="技术数据详情",
                        content=json.dumps(data_stats, ensure_ascii=False),
                        confidence_score=0.9,
                        metadata=json.dumps({"category": "data"}, ensure_ascii=False),
                    )
                )

    def _map_workflow_status(self, status: WorkflowStatus) -> AnalysisStatus:
        mapping = {
            WorkflowStatus.COMPLETED: AnalysisStatus.COMPLETED,
            WorkflowStatus.PENDING: AnalysisStatus.PENDING,
            WorkflowStatus.PROCESSING: AnalysisStatus.PROCESSING,
            WorkflowStatus.FAILED: AnalysisStatus.FAILED,
            WorkflowStatus.MAX_ITERATIONS: AnalysisStatus.MAX_ITERATIONS,
            WorkflowStatus.VALIDATION_FAILED: AnalysisStatus.VALIDATION_FAILED,
        }
        return mapping.get(status, AnalysisStatus.COMPLETED)

    def _build_result_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        def _serialize(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, "isoformat"):
                try:
                    return obj.isoformat()
                except Exception:  # pragma: no cover
                    return str(obj)
            if isinstance(obj, WorkflowStatus):
                return obj.value
            return obj

        payload = {
            "analysis_status": _serialize(state.get("analysis_status")),
            "flow_analysis": state.get("flow_analysis"),
            "llm_analysis": state.get("llm_analysis"),
            "data_stats": state.get("data_stats"),
            "retrieved_documents": state.get("retrieved_documents"),
            "final_report": state.get("final_report"),
            "refined_signals": state.get("refined_signals"),
            "signal_validation": state.get("signal_validation"),
            "timestamps": {
                "start": _serialize(state.get("start_time")),
                "end": _serialize(state.get("end_time")),
            },
        }
        return payload

    def _mark_failed(self, analysis_id: int, message: str) -> None:
        with session_scope() as session:
            analysis = session.get(ChargingAnalysis, analysis_id)
            if analysis is None:
                return
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = message
            analysis.completed_at = datetime.utcnow()
            session.add(analysis)


_analysis_service: AnalysisService | None = None


def get_analysis_service() -> AnalysisService:
    """提供单例服务实例。"""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service
