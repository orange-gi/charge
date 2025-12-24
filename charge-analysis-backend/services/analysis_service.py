"""充电分析调度与结果持久化服务。"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from database import session_scope
from langgraph_workflow import (
    ChargingAnalysisWorkflow,
    AnalysisStatus as WorkflowStatus,
    create_initial_workflow_trace,
)
from models import AnalysisResult, AnalysisStatus, ChargingAnalysis

logger = logging.getLogger(__name__)
class AnalysisService:
    """负责协调 LangGraph 工作流与数据库交互。"""

    def __init__(self) -> None:
        self.workflow = ChargingAnalysisWorkflow()
        # 存储正在运行的分析任务，用于取消操作
        self._running_tasks: Dict[int, asyncio.Task] = {}
        self._cancellation_flags: Dict[int, bool] = {}

    async def run_analysis(self, analysis_id: int, signal_names: list[str] | None = None) -> None:
        """执行工作流并将结果写入数据库。
        
        Args:
            analysis_id: 分析ID
            signal_names: 要解析的信号名称列表，如果为None则解析所有信号
        """
        logger.info("=" * 60)
        logger.info(f"开始执行分析任务 ID: {analysis_id}")
        logger.info(f"信号列表: {signal_names if signal_names else '全部信号'}")
        logger.info("=" * 60)
        
        # 检查是否已取消
        self._cancellation_flags[analysis_id] = False
        
        initial_state = self._build_initial_state(analysis_id, signal_names)
        if initial_state is None:
            logger.error("分析 %s 不存在，终止", analysis_id)
            return

        # 创建任务并存储引用
        task = asyncio.create_task(self._execute_analysis(analysis_id, initial_state))
        self._running_tasks[analysis_id] = task

        try:
            await task
        except asyncio.CancelledError:
            logger.warning(f"分析 {analysis_id} 被用户取消")
            self._mark_cancelled(analysis_id)
        finally:
            # 清理任务引用
            self._running_tasks.pop(analysis_id, None)
            self._cancellation_flags.pop(analysis_id, None)

    async def _execute_analysis(self, analysis_id: int, initial_state: Dict[str, Any]) -> None:
        """实际执行分析逻辑"""
        try:
            logger.info(f"执行分析工作流: {analysis_id}")
            final_state = await self.workflow.execute(initial_state)
            
            # 检查是否已取消
            if self._cancellation_flags.get(analysis_id, False):
                logger.warning(f"分析 {analysis_id} 在执行过程中被取消")
                return
            
            # 检查工作流是否失败
            workflow_status = final_state.get("analysis_status")
            if workflow_status == WorkflowStatus.FAILED:
                error_message = final_state.get("error_message", "工作流执行失败")
                logger.error(f"分析 {analysis_id} 工作流失败: {error_message}")
                self._mark_failed(analysis_id, error_message)
                return
                
            await self._persist_success_state(analysis_id, final_state)
            logger.info(f"分析 {analysis_id} 完成")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - 仅用于兜底
            logger.exception(f"分析 {analysis_id} 执行失败: {exc}")
            self._mark_failed(analysis_id, str(exc))

    def cancel_analysis(self, analysis_id: int) -> bool:
        """取消正在运行的分析任务
        
        无论任务是否存在，都会返回 True 表示操作成功。
        如果任务正在运行则取消它，如果没有任务则更新数据库状态。
        
        Returns:
            bool: 总是返回 True（操作成功）
        """
        cancelled_task = False
        
        # 如果有正在运行的任务，尝试取消它
        if analysis_id in self._running_tasks:
            logger.info(f"找到正在运行的分析任务: {analysis_id}，准备取消")
            self._cancellation_flags[analysis_id] = True
            task = self._running_tasks[analysis_id]
            if not task.done():
                task.cancel()
                cancelled_task = True
                logger.info(f"分析任务 {analysis_id} 已标记为取消")
            else:
                logger.info(f"分析任务 {analysis_id} 已完成或已停止")
        else:
            logger.info(f"分析任务 {analysis_id} 不在运行队列中")
        
        # 无论如何，都更新数据库状态为已取消
        # 这样即使任务不在运行，也能标记为已取消
        if not cancelled_task:
            logger.info(f"更新分析 {analysis_id} 的状态为已取消")
            self._mark_cancelled(analysis_id)
        
        return True  # 总是返回成功

    def _build_initial_state(self, analysis_id: int, signal_names: list[str] | None = None) -> Dict[str, Any] | None:
        # 在会话内提取所有需要的属性值，避免会话分离后的访问错误
        file_path_str: str | None = None
        file_size: int | None = None
        user_id: int | None = None
        created_at: datetime | None = None
        
        with session_scope() as session:
            analysis: ChargingAnalysis | None = session.get(ChargingAnalysis, analysis_id)
            if analysis is None:
                return None

            # 在会话内提取所有需要的属性值
            file_path_str = analysis.file_path
            file_size = analysis.file_size
            user_id = analysis.user_id
            created_at = analysis.created_at

            analysis.status = AnalysisStatus.PROCESSING
            analysis.progress = 5.0
            analysis.started_at = analysis.started_at or datetime.utcnow()

            session.add(analysis)

        # 在会话外使用已提取的值
        file_path = Path(file_path_str)
        progress_callback = self._progress_callback_factory(analysis_id)
        workflow_trace = create_initial_workflow_trace(file_path.name, file_size or 0, created_at)

        initial_state = {
            "analysis_id": str(analysis_id),
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_size or 0,
            "user_id": user_id,
            "validation_status": "pending",
            "parsing_status": "pending",
            "iteration": 0,
            "analysis_status": WorkflowStatus.PENDING,
            "progress_callback": progress_callback,
            "start_time": datetime.utcnow(),
            "selected_signals": signal_names,  # 添加用户选择的信号列表
            "workflow_trace": workflow_trace,
            "parsed_data_records": None,
            "raw_messages": None,  # 原始消息数据
        }
        logger.info(f"_build_initial_state: selected_signals={initial_state.get('selected_signals')}")
        return initial_state

    def _progress_callback_factory(self, analysis_id: int):
        async def _progress(stage: str, progress: float, message: str | None = None) -> None:
            # 检查是否已取消
            if self._cancellation_flags.get(analysis_id, False):
                logger.info(f"分析 {analysis_id} 进度更新被取消请求中断 (阶段: {stage})")
                return
                
            log_msg = f"分析 {analysis_id} 进度 {progress:.1f}% - {stage}"
            if message:
                log_msg += f": {message}"
            logger.info(log_msg)
            
            with session_scope() as session:
                analysis = session.get(ChargingAnalysis, analysis_id)
                if analysis is None:
                    return
                # 再次检查取消标志
                if self._cancellation_flags.get(analysis_id, False):
                    logger.debug(f"分析 {analysis_id} 在保存进度前被取消")
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
            # 保证写入数据库的是严格 JSON（禁止 NaN/Infinity）
            analysis.result_data = json.dumps(summary_payload, ensure_ascii=False, allow_nan=False)

            # 清理旧结果
            session.query(AnalysisResult).filter(
                AnalysisResult.analysis_id == analysis_id
            ).delete(synchronize_session=False)

            # 安全地获取最终报告摘要
            final_report = summary_payload.get("final_report") or {}
            summary_content = (
                llm_analysis.get("summary") 
                or (final_report.get("summary") if isinstance(final_report, dict) else "")
                or state.get("error_message", "分析完成")
            )
            
            # 安全地获取置信度分数
            flow_analysis = summary_payload.get("flow_analysis") or {}
            confidence_score = (
                flow_analysis.get("confidence") 
                if isinstance(flow_analysis, dict) 
                else 0.0
            )

            # 汇总结果
            session.add(
                AnalysisResult(
                    analysis_id=analysis_id,
                    result_type="summary",
                    title="充电分析总结",
                    content=summary_content,
                    confidence_score=confidence_score,
                    meta_info=json.dumps({"category": "overview", "risk": llm_analysis.get("risk_assessment")}, ensure_ascii=False),
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
                        meta_info=json.dumps({"category": "finding"}, ensure_ascii=False),
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
                        meta_info=json.dumps({"priority": "medium"}, ensure_ascii=False),
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
                        meta_info=json.dumps({"category": "data"}, ensure_ascii=False),
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
            """递归序列化对象，处理 numpy 和 pandas 类型"""
            # 处理 None
            if obj is None:
                return None

            # 处理 NaN / Infinity（Python json.dumps 默认 allow_nan=True 会生成 NaN/Infinity，前端 JSON.parse 会失败）
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
            
            # 处理 datetime
            if isinstance(obj, datetime):
                return obj.isoformat()
            
            # 处理 WorkflowStatus
            if isinstance(obj, WorkflowStatus):
                return obj.value
            
            # 处理 numpy 类型
            try:
                import numpy as np
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                                   np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                   np.uint32, np.uint64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                    val = float(obj)
                    if math.isnan(val) or math.isinf(val):
                        return None
                    return val
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except ImportError:
                pass  # numpy 未安装时跳过
            
            # 处理 pandas 类型
            try:
                import pandas as pd
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                if isinstance(obj, (pd.Series, pd.DataFrame)):
                    return obj.to_dict() if isinstance(obj, pd.Series) else obj.to_dict(orient='records')
            except ImportError:
                pass  # pandas 未安装时跳过
            
            # 处理字典 - 递归处理值
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            
            # 处理列表/元组 - 递归处理元素
            if isinstance(obj, (list, tuple)):
                return [_serialize(item) for item in obj]
            
            # 处理有 isoformat 方法的对象（如其他日期时间类型）
            if hasattr(obj, "isoformat"):
                try:
                    return obj.isoformat()
                except Exception:  # pragma: no cover
                    return str(obj)
            
            # 其他类型直接返回
            return obj

        payload = {
            "analysis_status": _serialize(state.get("analysis_status")),
            "validation_status": _serialize(state.get("validation_status")),
            "validation_message": _serialize(state.get("validation_message")),
            "parsing_status": _serialize(state.get("parsing_status")),
            "flow_analysis": _serialize(state.get("flow_analysis")),
            # 流程控制：关键信号规则处理产物
            "signal_windows": _serialize(state.get("signal_windows")),
            "signal_windows_list": _serialize(state.get("signal_windows_list")),
            "signal_rule_meta": _serialize(state.get("signal_rule_meta")),
            "rag_queries": _serialize(state.get("rag_queries")),
            "llm_analysis": _serialize(state.get("llm_analysis")),
            # 细化分析（循环）产物
            "refine_result": _serialize(state.get("refine_result")),
            "refine_confidence": _serialize(state.get("refine_confidence")),
            "additional_signals": _serialize(state.get("additional_signals")),
            "data_stats": _serialize(state.get("data_stats")),
            "parsed_data": _serialize(state.get("parsed_data_records")),
            "raw_messages": _serialize(state.get("raw_messages")),  # 添加原始消息数据
            "selected_signals": _serialize(state.get("selected_signals")),  # 保存选择的信号列表
            "retrieved_documents": _serialize(state.get("retrieved_documents")),
            "retrieval_by_query": _serialize(state.get("retrieval_by_query")),
            "final_report": _serialize(state.get("final_report")),
            "refined_signals": _serialize(state.get("refined_signals")),
            "signal_validation": _serialize(state.get("signal_validation")),
            "workflow_trace": _serialize(state.get("workflow_trace")),
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

    def _mark_cancelled(self, analysis_id: int) -> None:
        """标记分析为已取消"""
        with session_scope() as session:
            analysis = session.get(ChargingAnalysis, analysis_id)
            if analysis is None:
                return
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = "分析已被用户取消"
            analysis.completed_at = datetime.utcnow()
            session.add(analysis)


_analysis_service: AnalysisService | None = None


def get_analysis_service() -> AnalysisService:
    """提供单例服务实例。"""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service
