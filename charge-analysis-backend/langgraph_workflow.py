"""
LangGraph 工作流实现

包含所有 Agent 节点定义、工作流编排和状态管理。
"""

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from enum import Enum
import ipaddress
from urllib.parse import urlparse
import os

import pandas as pd
from langgraph.graph import StateGraph, START, END

# 设置日志
logger = logging.getLogger(__name__)


class AnalysisStatus(Enum):
    """分析状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATION_FAILED = "validation_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations_reached"


WORKFLOW_STEP_LABELS = {
    "file_upload": "文件上传",
    "file_validation": "文件验证",
    "message_parsing": "报文解析",
    "flow_control": "流程控制",
    "rag_retrieval": "RAG检索",
    "detailed_analysis": "细化分析",
    "llm_analysis": "LLM分析",
    "report_generation": "报告生成",
}


class ChargingAnalysisState(TypedDict):
    """充电分析工作流状态"""
    # 基础信息
    analysis_id: str
    file_path: str
    file_name: str
    file_size: int
    user_id: int
    
    # 文件状态
    validation_status: str
    parsing_status: str
    
    # 数据
    parsed_data: Optional[pd.DataFrame]
    data_stats: Optional[Dict[str, Any]]
    
    # 分析流程
    flow_analysis: Optional[Dict[str, Any]]
    problem_direction: Optional[str]
    confidence_score: Optional[float]

    # 流程控制：关键信号规则处理（用于循环细化）
    signals_to_process: Optional[List[str]]  # 本轮需要按“时间段->值”处理的信号
    processed_signals: Optional[List[str]]  # 已处理过的信号（规范化后小写）
    signal_windows: Optional[Dict[str, Dict[str, str]]]  # signal_name -> { "HH:MM:SS-HH:MM:SS": "value" }
    signal_windows_list: Optional[List[Dict[str, Dict[str, str]]]]  # 兼容前端：[{signal:{...}}, ...]
    signal_rule_meta: Optional[Dict[str, Any]]  # 枚举释义等元信息
    rag_queries: Optional[List[Dict[str, Any]]]  # [{signal,value,query,intervals:[...]}, ...]
    
    # RAG检索
    retrieved_documents: Optional[List[Dict[str, Any]]]
    retrieval_context: Optional[str]
    retrieval_status: Optional[str]
    retrieval_by_query: Optional[Dict[str, List[Dict[str, Any]]]]  # query -> filtered docs(score>=0.7)
    
    # 细化分析
    iteration: int
    refined_signals: Optional[List[str]]
    signal_validation: Optional[Dict[str, Any]]
    analysis_status: AnalysisStatus
    refine_result: Optional[Dict[str, Any]]  # {conclusion, confidence, needed_signals}
    refine_confidence: Optional[float]
    additional_signals: Optional[List[str]]  # 下一轮建议补充的信号（原始名称）
    
    # 最终结果
    llm_analysis: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    visualizations: Optional[List[Dict[str, Any]]]
    
    # 元数据
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str]
    progress_callback: Optional[callable]
    
    # 前端展示所需
    workflow_trace: Optional[Dict[str, Any]]
    parsed_data_records: Optional[List[Dict[str, Any]]]
    raw_messages: Optional[List[Dict[str, Any]]]  # 原始 CAN 消息数据
    selected_signals: Optional[List[str]]  # 用户选择的信号列表


def _iso_now() -> str:
    return datetime.utcnow().isoformat()


def _should_trust_env_proxies(base_url: str) -> bool:
    """内网 LLM 常见场景：应绕过 HTTP(S)_PROXY，避免代理导致 504/无法访问。"""
    # 显式开关：需要走代理时可设置 OPENAI_TRUST_ENV=1/true
    if str(os.environ.get("OPENAI_TRUST_ENV") or "").strip().lower() in {"1", "true", "yes", "y"}:
        return True
    try:
        host = urlparse(base_url or "").hostname or ""
    except Exception:
        host = ""
    if not host:
        # 没有 host 时保持默认：信任环境（不强行改变）
        return True
    host_l = host.lower()
    # 华为内网域名（你这里的 api.openai.rnd.huawei.com）默认认为是“应直连”
    # 说明：你报错页来自 HIS Proxy，意味着当前请求仍走了 SWG/代理；直连可避免 504。
    if host_l.endswith(".huawei.com") or host_l.endswith(".huawei.com.cn"):
        return False
    # 常见内网域名后缀
    if host_l.endswith(".internal") or host_l.endswith(".local") or host_l.endswith(".lan"):
        return False
    # 直连 IP：若是私网/回环/链路本地，则不信任环境代理
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except Exception:
        pass
    # 兜底：保持默认（信任环境）
    return True


def _ensure_no_proxy_for_base_url(base_url: str) -> None:
    """把 base_url 的 host 加入 NO_PROXY/no_proxy（注意：不要使用 *.domain 这种通配符）。"""
    try:
        host = urlparse(base_url or "").hostname or ""
    except Exception:
        host = ""
    if not host:
        return
    # 兼容不同库：NO_PROXY 与 no_proxy 都设置
    cur = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [p.strip() for p in cur.split(",") if p.strip()]
    # httpx/urllib3 通常支持 ".domain.com" 作为后缀匹配，但不保证支持 "*.domain.com"
    suffix = "." + ".".join(host.split(".")[-2:]) if host.count(".") >= 1 else ""
    for item in [host, suffix]:
        if item and item not in parts:
            parts.append(item)
    merged = ",".join(parts)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _ensure_trace_entry(state: ChargingAnalysisState, node_id: str) -> Dict[str, Any]:
    trace = state.setdefault("workflow_trace", {})
    entry = trace.get(node_id, {"node_id": node_id, "name": WORKFLOW_STEP_LABELS.get(node_id, node_id)})
    trace[node_id] = entry
    return entry


def mark_node_started(state: ChargingAnalysisState, node_id: str, description: str | None = None, metadata: Dict[str, Any] | None = None) -> None:
    entry = _ensure_trace_entry(state, node_id)
    now = _iso_now()
    # 支持同一节点多次执行（循环）：记录 runs 列表，前端可展示“细化分析→流程控制”的回环
    runs = entry.setdefault("runs", [])
    current_run = {"started_at": now, "status": "running"}
    if description:
        current_run["description"] = description
    if metadata:
        current_run["metadata"] = metadata
    entry["current_run"] = current_run
    # 兼容旧前端：仍保留顶层 started_at/ended_at/status/output 为“最后一次执行”
    entry["started_at"] = now
    entry["status"] = "running"
    if description:
        entry["description"] = description
    if metadata:
        entry["metadata"] = metadata


def mark_node_completed(state: ChargingAnalysisState, node_id: str, output: Dict[str, Any] | None = None) -> None:
    entry = _ensure_trace_entry(state, node_id)
    now = _iso_now()
    entry["status"] = "completed"
    entry["ended_at"] = now
    if output is not None:
        entry["output"] = output
    # 写入本次 run
    current = entry.pop("current_run", None)
    if isinstance(current, dict):
        current["status"] = "completed"
        current["ended_at"] = now
        if output is not None:
            current["output"] = output
        entry.setdefault("runs", []).append(current)


def mark_node_failed(state: ChargingAnalysisState, node_id: str, error_message: str) -> None:
    entry = _ensure_trace_entry(state, node_id)
    now = _iso_now()
    entry["status"] = "failed"
    entry["ended_at"] = now
    entry["error"] = error_message
    current = entry.pop("current_run", None)
    if isinstance(current, dict):
        current["status"] = "failed"
        current["ended_at"] = now
        current["error"] = error_message
        entry.setdefault("runs", []).append(current)


def create_initial_workflow_trace(file_name: str, file_size: int | None, created_at: datetime | None = None) -> Dict[str, Any]:
    """构建默认的节点追踪信息"""
    timestamp = (created_at or datetime.utcnow()).isoformat()
    trace: Dict[str, Any] = {
        "file_upload": {
            "node_id": "file_upload",
            "name": WORKFLOW_STEP_LABELS.get("file_upload"),
            "status": "completed",
            "started_at": timestamp,
            "ended_at": timestamp,
            "output": {
                "file_name": file_name,
                "file_size": file_size or 0,
            },
            "description": "用户上传原始日志文件",
        }
    }
    for node_id, node_name in WORKFLOW_STEP_LABELS.items():
        trace.setdefault(node_id, {
            "node_id": node_id,
            "name": node_name,
            "status": "pending",
        })
    return trace


class FileValidationNode:
    """文件验证节点"""
    
    def __init__(self):
        self.allowed_extensions = ['.blf', '.csv', '.xlsx']
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理文件验证"""
        file_path = state.get('file_path')
        file_name = state.get('file_name')
        file_size = state.get('file_size')
        mark_node_started(state, "file_validation", "验证文件合法性")
        
        try:
            # 验证文件类型
            if not any(file_name.endswith(ext) for ext in self.allowed_extensions):
                raise ValueError(f"不支持的文件类型: {file_name}")
            
            # 验证文件大小
            if file_size > self.max_file_size:
                raise ValueError(f"文件过大: {file_size / (1024*1024):.1f}MB")
            
            # 验证文件完整性
            if not await self._check_file_integrity(file_path):
                raise ValueError("文件损坏或格式错误")
            
            # 更新进度
            if state.get('progress_callback'):
                await state['progress_callback']("文件验证", 10, "文件验证通过")
            
            result_state = {
                **state,
                'validation_status': 'passed',
                'validation_message': '文件验证通过'
            }
            mark_node_completed(result_state, "file_validation", {
                "file_size": file_size,
                "file_name": file_name,
                "status": "passed"
            })
            return result_state
            
        except Exception as e:
            logger.error(f"文件验证失败: {str(e)}")
            if state.get('progress_callback'):
                await state['progress_callback']("文件验证", 0, f"文件验证失败: {str(e)}")
            
            failed_state = {
                **state,
                'validation_status': 'failed',
                'error_message': str(e),
                'analysis_status': AnalysisStatus.FAILED
            }
            mark_node_failed(failed_state, "file_validation", str(e))
            return failed_state
    
    async def _check_file_integrity(self, file_path: str) -> bool:
        """检查文件完整性"""
        try:
            import os
            return os.path.exists(file_path) and os.path.getsize(file_path) > 0
        except Exception:
            return False


class MessageParsingNode:
    """报文解析节点 - 基于 DBC 文件解析 CAN 日志"""
    
    def __init__(self):
        # 关键信号关键词（GBT 27930-2015 标准中的充电相关信号）
        # 注意：实际信号名称可能因 DBC 文件而异，这里使用通用关键词匹配；
        # 真正的 filter_signals 会在 process() 中基于“当前用户配置的 DBC”动态计算。
        self.filter_keywords = [
            'BMS', 'Chrg', 'SOC', 'Batt',      # 电池管理系统相关
            'Current', 'Voltage', 'Temp',      # 电气参数
            'State', 'Ready', 'Connect',       # 状态信息
            'Output', 'Max', 'Min',            # 输出参数
        ]
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理报文解析"""
        file_path = state['file_path']
        mark_node_started(state, "message_parsing", "解析 CAN 日志并提取信号")
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("报文解析", 15, "开始解析 CAN 日志...")

            # 解析当前用户使用的 DBC（如果未配置则回退到默认 DBC）
            user_id = int(state.get("user_id") or 0)
            from services.can_parser import CANLogParser
            from services.dbc_config import resolve_user_dbc_path

            dbc_path = resolve_user_dbc_path(user_id) if user_id else None
            can_parser = CANLogParser(dbc_file_path=dbc_path) if dbc_path else CANLogParser()
            logger.info(
                "报文解析准备完成 analysis_id=%s user_id=%s file=%s dbc=%s",
                state.get("analysis_id"),
                user_id,
                file_path,
                dbc_path or str(can_parser.dbc_file_path),
            )
            
            # 获取用户选择的信号列表，如果未指定则使用默认过滤信号
            selected_signals = state.get('selected_signals')
            logger.info(f"从 state 获取 selected_signals: {selected_signals} (type: {type(selected_signals)})")
            if selected_signals and len(selected_signals) > 0:
                # 使用用户选择的信号（仅解析这些信号，极大提升速度）
                filter_signals = selected_signals
                logger.info(f"使用用户选择的 {len(filter_signals)} 个信号进行解析: {filter_signals[:5]}{'...' if len(filter_signals) > 5 else ''}")
            else:
                # 使用默认的过滤信号列表（基于当前 DBC 动态匹配）
                all_signals = can_parser.get_available_signals()
                filter_signals = [
                    sig
                    for sig in all_signals
                    if any(keyword.lower() in sig.lower() for keyword in self.filter_keywords)
                ]
                logger.info(
                    "使用默认过滤信号进行解析 keyword_count=%s matched_signals=%s total_signals_in_dbc=%s",
                    len(self.filter_keywords),
                    len(filter_signals),
                    len(all_signals),
                )
                if not filter_signals:
                    logger.warning("默认关键词未匹配到任何信号，将解析全部信号（可能较慢）")
                    filter_signals = None
            
            # 检查文件类型
            file_ext = Path(file_path).suffix.lower()
            
            # 收集原始消息数据（用于前端展示）
            raw_messages = []
            if file_ext == '.blf':
                try:
                    if state.get('progress_callback'):
                        await state['progress_callback']("报文解析", 12, "收集原始消息数据...")
                    raw_messages = await can_parser.collect_raw_messages(file_path, max_messages=10000)
                    logger.info(f"收集到 {len(raw_messages)} 条原始消息数据")
                except Exception as e:
                    logger.warning(f"收集原始消息数据失败: {e}，将跳过原始数据展示")
            
            if file_ext == '.blf':
                # 使用 CAN 解析器解析 BLF 文件（高性能版本）
                df_parsed = await can_parser.parse_blf(
                    file_path,
                    filter_signals=filter_signals,
                    progress_callback=state.get('progress_callback')
                )
            else:
                # 其他格式（CSV等）使用备用方法
                logger.warning(f"不支持的文件格式: {file_ext}，使用备用解析方法")
                df_parsed = await self._parse_file_fallback(file_path, filter_signals)
            
            if df_parsed.empty:
                logger.warning("解析出的 DataFrame 为空，这可能意味着没有匹配到 CAN 消息")
                # 不抛出异常，而是返回空数据，让工作流继续
                df_cleaned = df_parsed
                stats = {
                    'total_records': 0,
                    'time_range': {'start': '', 'end': ''},
                    'signal_stats': {},
                    'signal_count': len([col for col in df_parsed.columns if col not in {'timestamp', 'ts', 'time', 'Time', 'can_id', 'message_name', 'dlc'}])
                }
            else:
                # 数据预处理和清洗
                if state.get('progress_callback'):
                    await state['progress_callback']("报文解析", 85, "正在进行数据清洗...")
                
                logger.info(f"解析出 {len(df_parsed)} 条原始记录，{len(df_parsed.columns)} 个列")
                df_cleaned = await self._clean_data(df_parsed)
                logger.info(f"数据清洗后剩余 {len(df_cleaned)} 条记录")
                
                # 如果清洗后数据为空，使用原始数据的前1000条（避免数据全部丢失）
                if df_cleaned.empty and not df_parsed.empty:
                    logger.warning("数据清洗后为空，使用原始数据的前1000条记录")
                    df_cleaned = df_parsed.head(1000)
                
                # 如果用户选择了特定信号，先过滤数据（用于统计和显示）
                metadata_columns = {'timestamp', 'ts', 'time', 'Time', 'can_id', 'message_name', 'dlc'}
                if selected_signals and len(selected_signals) > 0:
                    # 过滤数据，只保留选择的信号列（保留元数据列）
                    columns_to_keep = [col for col in df_cleaned.columns if col in metadata_columns or col in selected_signals]
                    df_for_display = df_cleaned[columns_to_keep]
                    logger.info(f"信号过滤：保留 {len(columns_to_keep)} 列（{len(selected_signals)} 个选择的信号 + {len(columns_to_keep) - len(selected_signals)} 个元数据列）")
                    logger.debug(f"保留的列：{columns_to_keep}")
                else:
                    df_for_display = df_cleaned
                
                # 提取关键统计信息（使用过滤后的数据）
                stats = await self._extract_statistics(df_for_display)
            
            # 限制保存的记录数量，避免数据过大（最多保存10000条）
            max_records = 10000
            if len(df_for_display) > max_records:
                logger.info(f"解析数据有 {len(df_for_display)} 条记录，限制为前 {max_records} 条用于前端展示")
                df_for_records = df_for_display.head(max_records)
            else:
                df_for_records = df_for_display
                
            parsed_records = df_for_records.to_dict(orient='records')
            
            # 确保 parsed_records 可以被序列化（处理 pandas Timestamp 等类型）
            import pandas as pd
            serializable_records = []
            for record in parsed_records:
                serializable_record = {}
                for k, v in record.items():
                    if pd.isna(v):
                        serializable_record[k] = None
                    elif isinstance(v, pd.Timestamp):
                        serializable_record[k] = v.isoformat()
                    elif isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                        serializable_record[k] = None
                    else:
                        serializable_record[k] = v
                serializable_records.append(serializable_record)
            
            result_state = {
                **state,
                'parsed_data': df_for_display,  # 使用过滤后的数据
                'parsed_data_records': serializable_records,
                'raw_messages': raw_messages,  # 添加原始消息数据
                'data_stats': stats,
                'parsing_status': 'completed',
                'selected_signals': selected_signals if selected_signals else None,  # 保存选择的信号列表
                'dbc_signals': can_parser.get_available_signals(),
                'dbc_file_used': dbc_path or str(can_parser.dbc_file_path),
            }
            mark_node_completed(result_state, "message_parsing", {
                "records": len(serializable_records),
                "total_parsed_records": len(df_cleaned),  # 实际解析的总记录数
                "signal_count": stats.get('signal_count', 0),
                "dbc": result_state.get("dbc_file_used"),
            })
            return result_state
            
        except Exception as e:
            logger.error(f"报文解析失败: {str(e)}", exc_info=True)
            if state.get('progress_callback'):
                await state['progress_callback']("报文解析", 0, f"报文解析失败: {str(e)}")
            
            failed_state = {
                **state,
                'parsing_status': 'failed',
                'error_message': str(e),
                'analysis_status': AnalysisStatus.FAILED
            }
            mark_node_failed(failed_state, "message_parsing", str(e))
            return failed_state
    
    async def _parse_file_fallback(self, file_path: str, filter_signals: List[str]) -> pd.DataFrame:
        """备用文件解析方法（用于非 BLF 格式）"""
        logger.info(f"使用备用方法解析文件: {file_path}")
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            # 尝试转换时间戳列
            for col in ['timestamp', 'ts', 'time', 'Time']:
                if col in df.columns:
                    df['ts'] = pd.to_datetime(df[col])
                    break
            return df
        else:
            # 其他格式暂不支持
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .blf 和 .csv 格式")
    
    async def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗，基于实际数据列"""
        if df.empty:
            logger.warning("输入 DataFrame 为空，跳过数据清洗")
            return df
            
        df_cleaned = df.copy()
        
        # 排除元数据列
        metadata_columns = {'timestamp', 'ts', 'time', 'Time', 'can_id', 'message_name', 'dlc'}
        signal_columns = [col for col in df_cleaned.columns if col not in metadata_columns]
        
        if not signal_columns:
            logger.warning("未找到信号列，返回原始数据")
            return df_cleaned
        
        # 移除所有信号列都为空的行（保留至少有一个信号有值的行）
        if signal_columns:
            # 只删除所有信号列都为空的行
            df_cleaned = df_cleaned.dropna(subset=signal_columns, how='all')
        
        if df_cleaned.empty:
            logger.warning("数据清洗后 DataFrame 为空，返回原始数据")
            return df  # 返回原始数据，避免数据全部丢失
        
        # 对数值型信号进行异常值处理（不删除行，只标记或限制值）
        # 注意：这里不删除行，而是将异常值替换为边界值，避免数据丢失
        for col in signal_columns:
            if col not in df_cleaned.columns:
                continue
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                try:
                    # 计算分位数，但只在有足够数据时进行
                    non_null_values = df_cleaned[col].dropna()
                    if len(non_null_values) < 4:  # 至少需要4个值才能计算IQR
                        continue
                    
                    Q1 = non_null_values.quantile(0.25)
                    Q3 = non_null_values.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # 如果 IQR 为 0（所有值相同），跳过过滤
                    if IQR == 0:
                        continue
                    
                    lower_bound = Q1 - 10 * IQR  # 使用10倍IQR以保留更多数据
                    upper_bound = Q3 + 10 * IQR
                    
                    # 不删除行，而是将极端异常值替换为边界值（保留数据用于展示）
                    # 如果数据确实有严重异常，可以在前端展示时处理
                    # 这里只处理极端的无穷大和负无穷大
                    df_cleaned[col] = df_cleaned[col].replace([float('inf'), float('-inf')], [upper_bound, lower_bound])
                    
                except Exception as e:
                    logger.warning(f"处理信号 {col} 的异常值时出错: {e}，跳过该列")
                    continue
        
        logger.info(f"数据清洗完成: 原始 {len(df)} 条 -> 清洗后 {len(df_cleaned)} 条")
        return df_cleaned
    
    async def _extract_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """提取统计信息，基于实际解析的信号列"""
        # 排除元数据列
        metadata_columns = {'timestamp', 'ts', 'time', 'Time', 'can_id', 'message_name', 'dlc'}
        signal_columns = [col for col in df.columns if col not in metadata_columns]
        
        signal_stats = {}
        for col in signal_columns:
            if col not in df.columns:
                continue
                
            try:
                # 尝试数值统计
                if pd.api.types.is_numeric_dtype(df[col]):
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        signal_stats[col] = {
                            'mean': float(non_null.mean()) if not math.isnan(non_null.mean()) else None,
                            'std': float(non_null.std()) if not math.isnan(non_null.std()) else None,
                            'min': float(non_null.min()) if not math.isnan(non_null.min()) else None,
                            'max': float(non_null.max()) if not math.isnan(non_null.max()) else None,
                            'type': 'numeric'
                        }
                    else:
                        signal_stats[col] = {
                            'mean': None,
                            'std': None,
                            'min': None,
                            'max': None,
                            'type': 'numeric'
                        }
                else:
                    # 分类统计
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 20:  # 只对类别数较少的列统计分布
                        signal_stats[col] = {
                            'unique_values': unique_vals.tolist() if len(unique_vals) > 0 else [],
                            'distribution': df[col].value_counts().to_dict() if len(df[col].dropna()) > 0 else {},
                            'type': 'categorical'
                        }
                    else:
                        signal_stats[col] = {
                            'unique_count': len(unique_vals),
                            'type': 'categorical_many'
                        }
            except Exception as e:
                logger.warning(f"提取信号 {col} 统计信息失败: {e}")
                continue
        
        # 安全地提取时间范围
        time_start = ''
        time_end = ''
        if 'ts' in df.columns and len(df) > 0:
            try:
                ts_min = df['ts'].min()
                ts_max = df['ts'].max()
                if pd.notna(ts_min):
                    time_start = ts_min.isoformat() if hasattr(ts_min, 'isoformat') else str(ts_min)
                if pd.notna(ts_max):
                    time_end = ts_max.isoformat() if hasattr(ts_max, 'isoformat') else str(ts_max)
            except Exception as e:
                logger.warning(f"提取时间范围失败: {e}")
        
        stats = {
            'total_records': len(df),
            'time_range': {
                'start': time_start,
                'end': time_end
            },
            'signal_stats': signal_stats,
            'signal_count': len(signal_columns)
        }
        
        return stats


class FlowControlModelNode:
    """流程控制模型节点"""

    _KEY_SIGNALS = [
        "BMS_DCChrgSt",
        "BMS_ChrgEndNum",
        "BMS_FaultNum1",
        "VIU0_FaultNum1",
        "CHM_ComVersion",
    ]

    # 流程控制阶段认为“无效/占位”的值：解析出来后直接过滤，不报错
    # 备注：用户侧提到 15777215；此前也出现过 16777215（常见 0xFFFFFF）
    _INVALID_VALUES = {"0", "16777215", "15777215"}

    _DCCHRGST_ENUM = {
        0x0: "Off 未插枪",
        0x1: "Init 初始化",
        0x2: "Standby 待机",
        0x3: "Charging 充电中",
        0x4: "CHGEnd 充电终止",
        0x5: "PreHeat 预热",
        0x6: "Fault 充电异常终止",
        0x7: "Schedule 预约",
        0x8: "Power supply mode 供电模式",
        0x9: "EmergencyStop 紧急结束",
    }

    _CHM_COMVERSION_ENUM = {
        257: "2015国标桩",
        16777215: "无效值",
        5456689: "湾区协议",
        2313: "特来电",
        480: "蔚来私有桩",
    }

    _TIME_COLUMNS = ["ts", "timestamp", "time", "Time"]

    def _normalize_name(self, name: str) -> str:
        return (name or "").strip().lower()

    def _find_column_case_insensitive(self, df: pd.DataFrame, target: str) -> str | None:
        want = self._normalize_name(target)
        for col in df.columns:
            if self._normalize_name(str(col)) == want:
                return str(col)
        return None

    def _coerce_int(self, v: Any) -> int | None:
        if v is None:
            return None
        try:
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, (int,)):
                return int(v)
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
                return int(v)
            s = str(v).strip()
            if not s:
                return None
            if s.lower().startswith("0x"):
                return int(s, 16)
            # 有些解析器会输出 "1.0"
            if "." in s:
                f = float(s)
                if math.isnan(f) or math.isinf(f):
                    return None
                return int(f)
            return int(s)
        except Exception:
            return None

    def _format_hms(self, ts: Any) -> str:
        try:
            t = pd.to_datetime(ts, errors="coerce")
            if pd.isna(t):
                return ""
            # 只保留 HH:MM:SS
            return t.strftime("%H:%M:%S")
        except Exception:
            return ""

    def _get_time_series(self, df: pd.DataFrame) -> pd.Series | None:
        for col in self._TIME_COLUMNS:
            if col in df.columns:
                return df[col]
        return None

    def _build_value_windows(self, df: pd.DataFrame, signal_col: str) -> Dict[str, str]:
        """将信号序列切分为时间段->值（字符串）。解析不出来就返回空 dict，不报错。"""
        if df is None or df.empty or signal_col not in df.columns:
            return {}
        ts_series = self._get_time_series(df)
        if ts_series is None:
            return {}

        # 只取时间与该信号，按时间排序
        tmp = pd.DataFrame({"_ts": ts_series, "_v": df[signal_col]})
        tmp["_ts"] = pd.to_datetime(tmp["_ts"], errors="coerce")
        tmp = tmp.dropna(subset=["_ts"])
        if tmp.empty:
            return {}
        tmp = tmp.sort_values("_ts", kind="mergesort")
        # 丢掉空值
        tmp = tmp.dropna(subset=["_v"])
        if tmp.empty:
            return {}

        windows: Dict[str, str] = {}
        run_start = tmp.iloc[0]["_ts"]
        last_ts = tmp.iloc[0]["_ts"]
        last_raw = tmp.iloc[0]["_v"]

        def _val_str(x: Any) -> str:
            iv = self._coerce_int(x)
            if iv is None:
                # 保底：原样字符串
                return str(x)
            return str(iv)

        last_val = _val_str(last_raw)
        for i in range(1, len(tmp)):
            row = tmp.iloc[i]
            cur_ts = row["_ts"]
            cur_val = _val_str(row["_v"])
            if cur_val != last_val:
                start_s = self._format_hms(run_start)
                end_s = self._format_hms(cur_ts)
                if start_s and end_s:
                    windows[f"{start_s}-{end_s}"] = last_val
                run_start = cur_ts
                last_val = cur_val
            last_ts = cur_ts

        # 末段
        start_s = self._format_hms(run_start)
        end_s = self._format_hms(last_ts)
        if start_s and end_s:
            windows[f"{start_s}-{end_s}"] = last_val
        # 过滤无效值（能解析多少算多少）
        try:
            windows = {k: v for k, v in windows.items() if str(v) not in self._INVALID_VALUES}
        except Exception:
            pass
        return windows

    def _build_rag_query(self, signal: str, value: str) -> str:
        # 针对你给的知识库示例做一点“更容易命中”的模板
        norm = self._normalize_name(signal)
        if norm == "bms_chrgendnum":
            return f"停充码ChrgEndNum 8bit={value}"
        # 其余信号先用通用格式
        return f"{signal}={value}"

    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理流程控制：关键信号规则化 + 构建 RAG 查询。"""
        df = state.get('parsed_data')
        stats = state.get('data_stats')
        iteration = int(state.get("iteration") or 0)
        mark_node_started(state, "flow_control", f"流程控制规则处理（第{iteration + 1}轮）", {"iteration": iteration})
        
        if df is None or stats is None:
            failed_state = {**state, "error_message": "缺少解析数据"}
            mark_node_failed(failed_state, "flow_control", "缺少解析数据")
            return failed_state
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("流程分析", 50, "流程控制模型分析中...")

            # 本轮要处理哪些信号：优先 signals_to_process（循环时由细化分析给出），否则默认5个关键
            signals_to_process = state.get("signals_to_process") or list(self._KEY_SIGNALS)
            processed = set(state.get("processed_signals") or [])

            signal_windows: Dict[str, Dict[str, str]] = dict(state.get("signal_windows") or {})
            signal_windows_list: List[Dict[str, Dict[str, str]]] = list(state.get("signal_windows_list") or [])
            rule_meta: Dict[str, Any] = dict(state.get("signal_rule_meta") or {})

            # 生成 windows + 枚举释义
            newly_processed_norm: List[str] = []
            for sig in signals_to_process:
                if not sig:
                    continue
                sig_norm = self._normalize_name(sig)
                if sig_norm in processed:
                    continue
                col = self._find_column_case_insensitive(df, sig)
                if not col:
                    continue
                windows = self._build_value_windows(df, col)
                if not windows:
                    continue
                # 使用“实际列名”作为 key（便于前端对齐 DataFrame 列）
                signal_windows[col] = windows
                signal_windows_list.append({col: windows})
                newly_processed_norm.append(sig_norm)

                # 枚举释义（可选，用于 LLM/前端展示）
                if sig_norm == "dcchrgst":
                    rule_meta[col] = {"enum": {str(k): v for k, v in self._DCCHRGST_ENUM.items()}}
                elif sig_norm == "bms_dcchrgst":
                    rule_meta[col] = {"enum": {str(k): v for k, v in self._DCCHRGST_ENUM.items()}}
                elif sig_norm == "chm_comversion":
                    rule_meta[col] = {"enum": {str(k): v for k, v in self._CHM_COMVERSION_ENUM.items()}, "default": "未知充电桩"}

            processed.update(newly_processed_norm)

            # 构建 rag_queries（按“信号-值”去重，避免一个信号很多段导致爆炸）
            rag_queries: List[Dict[str, Any]] = []
            for sig_name, windows in signal_windows.items():
                # 只针对本轮新增/关键处理的信号做查询：这里选择对所有已处理信号都可查询，但做去重
                value_to_intervals: Dict[str, List[str]] = {}
                for interval, val in (windows or {}).items():
                    # 再次过滤无效值，避免后续误查询
                    if str(val) in self._INVALID_VALUES:
                        continue
                    value_to_intervals.setdefault(str(val), []).append(str(interval))
                for val, intervals in value_to_intervals.items():
                    q = self._build_rag_query(sig_name, val)
                    rag_queries.append({"signal": sig_name, "value": str(val), "query": q, "intervals": intervals})

            # 仍保留原来 flow_analysis 的字段，避免前端/结果结构断裂
            analysis_result = {
                "handled_signal_count": len(signal_windows),
                "handled_signals": list(signal_windows.keys()),
                "iteration": iteration + 1,
            }

            if state.get('progress_callback'):
                await state['progress_callback']("流程分析", 60, f"流程分析完成（已规则化{len(signal_windows)}个信号）")

            result_state = {
                **state,
                "flow_analysis": analysis_result,
                "problem_direction": (state.get("problem_direction") or "general_analysis"),
                "confidence_score": float(state.get("confidence_score") or 0.0),
                "signals_to_process": signals_to_process,
                "processed_signals": sorted(processed),
                "signal_windows": signal_windows,
                "signal_windows_list": signal_windows_list,
                "signal_rule_meta": rule_meta,
                "rag_queries": rag_queries,
            }
            mark_node_completed(result_state, "flow_control", {
                "iteration": iteration + 1,
                "signals_requested": len(signals_to_process),
                "signals_processed_total": len(signal_windows),
                "rag_query_count": len(rag_queries),
            })
            return result_state
            
        except Exception as e:
            logger.error(f"流程控制处理失败: {str(e)}")
            failed_state = {
                **state,
                'error_message': str(e)
            }
            mark_node_failed(failed_state, "flow_control", str(e))
            return failed_state
    
    def _build_prompt(self, stats: Dict[str, Any]) -> str:
        """构建提示，基于实际解析的信号"""
        signal_stats = stats.get('signal_stats', {})
        
        # 构建信号统计描述
        signal_descriptions = []
        for signal_name, signal_data in signal_stats.items():
            if signal_data.get('type') == 'numeric':
                signal_descriptions.append(
                    f"- {signal_name}: 均值={signal_data.get('mean', 0):.2f}, "
                    f"标准差={signal_data.get('std', 0):.2f}, "
                    f"范围=[{signal_data.get('min', 0):.2f}, {signal_data.get('max', 0):.2f}]"
                )
            elif signal_data.get('type') == 'categorical':
                dist = signal_data.get('distribution', {})
                if dist:
                    signal_descriptions.append(
                        f"- {signal_name}: 分布={dist}"
                    )
        
        signals_text = "\n".join(signal_descriptions[:10])  # 限制显示前10个信号
        if len(signal_descriptions) > 10:
            signals_text += f"\n... 还有 {len(signal_descriptions) - 10} 个信号"
        
        prompt = f"""
充电数据分析：
总记录数: {stats['total_records']}
解析信号数: {stats.get('signal_count', 0)}
时间范围: {stats['time_range']['start']} 至 {stats['time_range']['end']}

信号统计：
{signals_text}

请根据这些数据分析可能的问题方向。输出格式为JSON。
"""
        return prompt
    
    async def _model_inference(self, prompt: str) -> str:
        """模拟模型推理"""
        await asyncio.sleep(1)  # 模拟推理时间
        
        # 返回模拟结果
        return '{"problem_direction": "charging_current_anomaly", "confidence": 0.78, "reasoning": "检测到充电电流异常模式"}'
    
    def _parse_model_output(self, output: str) -> Dict[str, Any]:
        """解析模型输出"""
        try:
            # 提取JSON部分
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = output[json_start:json_end]
                result = json.loads(json_str)
                return result
        except json.JSONDecodeError:
            pass
        
        # 备用解析逻辑
        return {
            'problem_direction': 'general_analysis',
            'confidence': 0.5,
            'reasoning': '模型输出解析失败，使用默认分析'
        }


class RAGRetrievalNode:
    """RAG检索节点"""
    
    def __init__(self):
        # 复用后端真实 RAG 服务（Chroma 检索）
        from services.rag_service import get_rag_service

        self._rag_service = get_rag_service()
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理RAG检索"""
        problem_direction = state.get('problem_direction', '')
        df = state.get('parsed_data')
        rag_queries = state.get("rag_queries") or []
        iteration = int(state.get("iteration") or 0)
        mark_node_started(state, "rag_retrieval", f"检索知识库（第{iteration + 1}轮）", {"iteration": iteration, "query_count": len(rag_queries)})
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("知识检索", 70, "RAG知识检索中...")
            
            # 优先使用流程控制生成的多查询；否则回退到旧逻辑（单查询）
            queries: List[str] = []
            if rag_queries:
                queries = [str(item.get("query") or "") for item in rag_queries if str(item.get("query") or "").strip()]
            else:
                query = self._build_retrieval_query(problem_direction, df)
                queries = [query]

            # 选择默认知识库（与 RAG 管理页一致：优先用户库，否则公共库）
            user_id = state.get("user_id")
            if not user_id:
                raise ValueError("缺少 user_id，无法选择知识库")

            collection = await asyncio.to_thread(self._rag_service.get_or_create_default_collection_for_user, int(user_id))
            retrieval_by_query: Dict[str, List[Dict[str, Any]]] = {}
            all_documents: List[Dict[str, Any]] = []
            # 阈值：只保留相似度 >= 0.7
            threshold = 0.7

            for q in queries:
                if not q:
                    continue
                result = await asyncio.to_thread(
                    self._rag_service.query,
                    collection.id,
                    q,
                    int(user_id),
                    5,
                    True,
                )
                docs = result.get("documents") or []
                filtered = [d for d in docs if float(d.get("score") or 0.0) >= threshold]
                retrieval_by_query[q] = filtered
                all_documents.extend(filtered)

            context = self._build_context(all_documents)
            
            if state.get('progress_callback'):
                await state['progress_callback']("知识检索", 80, "知识检索完成")
            
            result_state = {
                **state,
                'retrieved_documents': all_documents,
                'retrieval_context': context,
                'retrieval_status': 'completed',
                "retrieval_by_query": retrieval_by_query,
            }
            mark_node_completed(result_state, "rag_retrieval", {
                "document_count": len(all_documents),
                "query_count": len([q for q in queries if q]),
                "threshold": threshold,
            })
            return result_state
            
        except Exception as e:
            logger.error(f"RAG检索失败: {str(e)}")
            failed_state = {
                **state,
                'retrieval_status': 'failed',
                'error_message': str(e)
            }
            mark_node_failed(failed_state, "rag_retrieval", str(e))
            return failed_state
    
    def _build_retrieval_query(self, problem_direction: str, df: pd.DataFrame) -> str:
        """构建检索查询，基于实际数据特征"""
        base_query = f"充电系统{problem_direction}"
        
        # 基于实际数据特征调整查询
        # 查找可能的状态信号（包含 status, state, st 等关键词）
        status_signals = [col for col in df.columns if any(keyword in col.lower() for keyword in ['status', 'state', 'st'])]
        
        if status_signals:
            # 使用第一个找到的状态信号
            status_col = status_signals[0]
            try:
                status_dist = df[status_col].value_counts().to_dict()
                # 根据状态分布添加查询关键词
                if status_dist:
                    base_query += f" {status_col}异常"
            except Exception:
                pass
        
        return base_query
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """构建上下文"""
        context_parts = []
        for doc in documents:
            score = doc.get("score") or 0.0
            content = doc.get("content") or doc.get("snippet") or ""
            meta = doc.get("metadata") or {}
            src = doc.get("filename") or meta.get("source_filename") or "未知来源"
            row = doc.get("row_index") or meta.get("row_index")
            row_part = f"@{row}" if row is not None else ""
            context_parts.append(f"[{float(score):.3f}] {src}{row_part} {content}")
        return "\n\n".join(context_parts)


class DetailedAnalysisNode:
    """细化分析节点"""
    
    def __init__(self):
        self.max_iterations = 3
        # 复用 OpenAI 配置（有则用，无则 mock）
        from config import get_settings
        settings = get_settings()
        self._model_name = getattr(settings, "llm_model_name", None)
        self._llm_client = None
        if getattr(settings, "openai_api_key", None) and getattr(settings, "openai_base_url", None):
            try:
                from openai import AsyncOpenAI
                base_url = str(settings.openai_base_url)
                _ensure_no_proxy_for_base_url(base_url)
                trust_env = _should_trust_env_proxies(base_url)
                # 不同 openai SDK 版本对 http_client 类型要求不同，这里优先使用 httpx.AsyncClient（兼容性最好）
                http_client = None
                try:
                    import httpx  # type: ignore
                    http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0), trust_env=trust_env)
                except Exception:
                    http_client = None

                if http_client is not None:
                    self._llm_client = AsyncOpenAI(
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                        timeout=180.0,
                        max_retries=0,
                        http_client=http_client,
                    )
                else:
                    self._llm_client = AsyncOpenAI(
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                        timeout=180.0,
                        max_retries=0,
                    )
            except Exception:
                self._llm_client = None
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理细化分析"""
        problem_direction = state.get('problem_direction', '')
        context = state.get('retrieval_context', '')
        df = state.get('parsed_data')
        iteration = int(state.get('iteration', 0) or 0)
        mark_node_started(state, "detailed_analysis", f"细化分析（第{iteration + 1}轮）", {"iteration": iteration})
        
        if iteration >= self.max_iterations:
            if state.get('progress_callback'):
                await state['progress_callback']("细化分析", 85, "达到最大迭代次数")
            
            capped_state = {
                **state,
                'analysis_status': AnalysisStatus.MAX_ITERATIONS,
                'iteration': iteration
            }
            mark_node_completed(capped_state, "detailed_analysis", {
                "iteration": iteration,
                "status": "max_iterations"
            })
            return capped_state
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("细化分析", 85, f"细化分析（第{iteration+1}轮）")

            # 1) 先用“已有数据列+问题方向”给出一个基础候选信号集（兜底）
            refined_signals = await self._extract_refined_signals(problem_direction, context, df)
            signal_validation = await self._validate_signals(df, refined_signals)

            # 2) 把流程控制生成的“时间段->值”和 RAG 证据统一交给 LLM 细化分析，产出置信度与“需要更多哪些信号”
            windows_list = state.get("signal_windows_list") or []
            rule_meta = state.get("signal_rule_meta") or {}
            retrieval_by_query = state.get("retrieval_by_query") or {}
            refine = await self._llm_refine(problem_direction, windows_list, rule_meta, retrieval_by_query, signal_validation)
            refine_conf = float(refine.get("confidence") or 0.0)
            needed = refine.get("needed_signals") or []
            if not isinstance(needed, list):
                needed = []
            needed = [str(x) for x in needed if str(x).strip()]

            if state.get('progress_callback'):
                await state['progress_callback']("细化分析", 90, f"细化分析完成（置信度{refine_conf:.0%}）")

            # 3) 若置信度不足且未超过3轮，则准备下一轮回到 flow_control
            should_loop = refine_conf < 0.7 and (iteration + 1) < self.max_iterations and bool(needed)
            next_iteration = iteration + 1 if should_loop else iteration

            result_state = {
                **state,
                "refined_signals": refined_signals,
                "signal_validation": signal_validation,
                "refine_result": refine,
                "refine_confidence": refine_conf,
                "additional_signals": needed,
                "analysis_status": AnalysisStatus.PROCESSING if should_loop else AnalysisStatus.COMPLETED,
                "iteration": next_iteration,
                # 下一轮让 flow_control 按同样规则处理这些信号（大小写不敏感）
                "signals_to_process": needed if should_loop else (state.get("signals_to_process") or []),
            }
            mark_node_completed(result_state, "detailed_analysis", {
                "validated": bool(signal_validation.get("validated")),
                "refine_confidence": refine_conf,
                "will_loop": should_loop,
                "next_signal_count": len(needed) if should_loop else 0,
                "iteration": next_iteration + 1,
            })
            return result_state
            
        except Exception as e:
            logger.error(f"细化分析失败: {str(e)}")
            failed_state = {
                **state,
                'analysis_status': AnalysisStatus.VALIDATION_FAILED,
                'error_message': str(e)
            }
            mark_node_failed(failed_state, "detailed_analysis", str(e))
            return failed_state

    async def _llm_refine(
        self,
        direction: str,
        windows_list: List[Dict[str, Dict[str, str]]],
        rule_meta: Dict[str, Any],
        retrieval_by_query: Dict[str, List[Dict[str, Any]]],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM 细化分析：输出 conclusion/confidence/needed_signals。未配置时返回 mock。"""
        # mock
        if not self._llm_client or not self._model_name:
            # 兜底：给一个中等偏低置信度，并建议再补一些信号
            needed_signals = []
            try:
                # 从已规则化信号 windows 里推导（尽量不重复）
                seen = set()
                for item in windows_list[:5]:
                    for k in item.keys():
                        seen.add(str(k).lower())
                # 建议再补充一些常见电气量信号
                candidates = ["ChargeCurrent", "BatteryVoltage", "SOC", "BMS_State", "ConnectorState"]
                for c in candidates:
                    if c.lower() not in seen:
                        needed_signals.append(c)
            except Exception:
                needed_signals = ["ChargeCurrent", "BatteryVoltage", "SOC"]
            return {
                "conclusion": "基于当前关键信号与知识库证据，尚无法高置信度定性，需要补充更多信号细节。",
                "confidence": 0.55,
                "needed_signals": needed_signals[:5],
                "direction": direction or "general_analysis",
            }

        prompt = {
            "direction": direction or "general_analysis",
            "signal_windows": windows_list,
            "signal_rule_meta": rule_meta,
            "rag_evidence": retrieval_by_query,
            "signal_validation": validation or {},
            "requirements": {
                "confidence_threshold": 0.7,
                "max_iterations": self.max_iterations,
                "output_json_schema": {
                    "conclusion": "string",
                    "confidence": "number(0~1)",
                    "needed_signals": ["string", "..."],
                },
            },
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "你是充电系统诊断专家。"
                    "输入包含关键信号的时间段值变化、枚举释义与知识库检索证据。"
                    "请给出：1)当前阶段性结论(conclusion) 2)置信度(confidence 0~1) "
                    "3)为了更好定性问题还需要哪些信号明细(needed_signals)。"
                    "必须返回严格 JSON 对象，字段仅包含 conclusion/confidence/needed_signals。"
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        try:
            # 使用 stream，避免网关长连接读超时（504）
            stream = await self._llm_client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
                stream=True,
            )
            chunks: list[str] = []
            async for event in stream:
                try:
                    delta = event.choices[0].delta
                    if delta and delta.content:
                        chunks.append(delta.content)
                except Exception:
                    continue
            content = "".join(chunks) or ""
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                payload = json.loads(content[json_start:json_end])
            else:
                payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("LLM 返回非 JSON 对象")
            return {
                "conclusion": str(payload.get("conclusion") or ""),
                "confidence": float(payload.get("confidence") or 0.0),
                "needed_signals": payload.get("needed_signals") or [],
                "direction": direction or "general_analysis",
            }
        except Exception as exc:
            # 尽可能打印 HTTP 状态/响应体片段，便于定位 504 来自哪层网关
            try:
                from openai import APIStatusError
                if isinstance(exc, APIStatusError):
                    body = ""
                    try:
                        body = (exc.response.text or "")[:800]
                    except Exception:
                        body = ""
                    logger.warning(
                        "细化分析 LLM APIStatusError status=%s body=%s",
                        getattr(exc, "status_code", None),
                        body,
                        exc_info=True,
                    )
                else:
                    logger.warning("细化分析 LLM 调用失败，将使用 mock: %s", exc, exc_info=True)
            except Exception:
                logger.warning("细化分析 LLM 调用失败，将使用 mock: %s", exc, exc_info=True)
            return {
                "conclusion": "LLM 细化分析失败，使用兜底策略。",
                "confidence": 0.5,
                "needed_signals": ["ChargeCurrent", "BatteryVoltage", "SOC"],
                "direction": direction or "general_analysis",
            }
    
    async def _extract_refined_signals(self, direction: str, context: str, df: pd.DataFrame) -> List[str]:
        """提取细化信号，基于实际数据列和问题方向"""
        # 排除元数据列
        metadata_columns = {'timestamp', 'ts', 'time', 'Time', 'can_id', 'message_name', 'dlc'}
        available_columns = [col for col in df.columns if col not in metadata_columns]
        
        # 根据问题方向关键词匹配相关信号
        direction_lower = direction.lower()
        relevant_signals = []
        
        # 根据问题方向的关键词匹配信号名称
        keywords_mapping = {
            'connection': ['connect', 'link', 'attach'],
            'current': ['current', 'currt', 'ampere'],
            'voltage': ['voltage', 'volt', 'vlt'],
            'soc': ['soc', 'state_of_charge', 'battery_level'],
            'temperature': ['temp', 'temperature', 'thermal'],
            'status': ['status', 'state', 'st']
        }
        
        for keyword, patterns in keywords_mapping.items():
            if keyword in direction_lower:
                # 查找包含这些关键词的信号
                matched = [col for col in available_columns 
                          if any(pattern.lower() in col.lower() for pattern in patterns)]
                relevant_signals.extend(matched)
        
        # 如果没有匹配到，返回前几个可用的信号
        if not relevant_signals:
            relevant_signals = available_columns[:5]  # 返回前5个信号
        
        return list(set(relevant_signals))
    
    async def _validate_signals(self, df: pd.DataFrame, signals: List[str]) -> Dict[str, Any]:
        """验证信号"""
        validation_results = []
        for signal in signals:
            if signal in df.columns:
                quality_score = self._calculate_signal_quality(df[signal])
                validation_results.append({
                    'signal': signal,
                    'available': True,
                    'quality_score': quality_score,
                    'data_points': len(df[signal].dropna())
                })
            else:
                validation_results.append({
                    'signal': signal,
                    'available': False,
                    'quality_score': 0.0,
                    'data_points': 0
                })
        
        available_signals = [r for r in validation_results if r['available']]
        avg_quality = sum(r['quality_score'] for r in available_signals) / len(available_signals) if available_signals else 0
        
        return {
            'validated': len(available_signals) >= 2 and avg_quality > 0.6,
            'quality_threshold': 0.6,
            'signal_count': len(available_signals),
            'results': validation_results,
            'average_quality': avg_quality
        }
    
    def _calculate_signal_quality(self, series: pd.Series) -> float:
        """计算信号质量分数"""
        non_null_ratio = series.count() / len(series)
        variance = series.var()
        
        quality = non_null_ratio * 0.7
        if variance > 0 and variance < 1000:
            quality += 0.3
        elif variance == 0:
            quality += 0.1
        
        return min(quality, 1.0)


class LLMAnalysisNode:
    """LLM分析节点"""
    
    def __init__(self):
        from config import get_settings
        settings = get_settings()
        
        # 如果配置了 API Key 和 Base URL，则初始化 OpenAI 客户端
        if settings.openai_api_key and settings.openai_base_url:
            try:
                from openai import AsyncOpenAI
                base_url = str(settings.openai_base_url)
                _ensure_no_proxy_for_base_url(base_url)
                trust_env = _should_trust_env_proxies(base_url)
                http_client = None
                try:
                    import httpx  # type: ignore
                    http_client = httpx.AsyncClient(timeout=httpx.Timeout(240.0), trust_env=trust_env)
                except Exception:
                    http_client = None

                if http_client is not None:
                    self.llm_client = AsyncOpenAI(
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                        timeout=240.0,
                        max_retries=0,
                        http_client=http_client,
                    )
                else:
                    self.llm_client = AsyncOpenAI(
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                        timeout=240.0,
                        max_retries=0,
                    )
                self.model_name = settings.llm_model_name
                logger.info(f"LLM 客户端已初始化: model={self.model_name}, base_url={settings.openai_base_url}")
            except ImportError:
                logger.warning("openai 库未安装，将使用模拟 LLM 分析")
                self.llm_client = None
                self.model_name = None
            except Exception as e:
                logger.error(f"初始化 LLM 客户端失败: {e}，将使用模拟 LLM 分析")
                self.llm_client = None
                self.model_name = None
        else:
            logger.info("未配置 OPENAI_API_KEY 或 OPENAI_BASE_URL，将使用模拟 LLM 分析")
            self.llm_client = None
            self.model_name = None
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理LLM分析"""
        problem_direction = state.get('problem_direction')
        context = state.get('retrieval_context', '')
        data_stats = state.get('data_stats', {})
        refined_signals = state.get('refined_signals', [])
        validation = state.get('signal_validation', {})
        refine_result = state.get("refine_result") or {}
        signal_windows_list = state.get("signal_windows_list") or []
        rule_meta = state.get("signal_rule_meta") or {}
        retrieval_by_query = state.get("retrieval_by_query") or {}
        mark_node_started(state, "llm_analysis", "生成诊断结论")
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("LLM分析", 95, "LLM生成分析报告...")
            
            # 构建分析提示（安全处理可能为 None 的值）
            problem_direction = problem_direction or 'general_analysis'
            prompt = self._build_analysis_prompt(
                problem_direction,
                context,
                data_stats or {},
                refined_signals or [],
                validation or {},
                refine_result,
                signal_windows_list,
                rule_meta,
                retrieval_by_query,
            )
            
            # 模拟LLM分析
            analysis_result = await self._llm_analysis(prompt)
            
            if state.get('progress_callback'):
                await state['progress_callback']("LLM分析", 100, "分析报告生成完成")
            
            result_state = {
                **state,
                'llm_analysis': analysis_result,
                'analysis_status': AnalysisStatus.COMPLETED
            }
            mark_node_completed(result_state, "llm_analysis", {
                "summary": analysis_result.get("summary"),
                "risk": analysis_result.get("risk_assessment")
            })
            return result_state
            
        except Exception as e:
            logger.error(f"LLM分析失败: {str(e)}")
            failed_state = {
                **state,
                'llm_analysis': {
                    "summary": "分析过程中发生错误",
                    "findings": [],
                    "risk_assessment": "未知",
                    "recommendations": ["请联系技术支持"],
                    "technical_details": str(e)
                },
                'error_message': str(e)
            }
            mark_node_failed(failed_state, "llm_analysis", str(e))
            return failed_state
    
    def _build_analysis_prompt(
        self,
        direction: str,
        context: str,
        stats: Dict,
        signals: List[str],
        validation: Dict,
        refine_result: Dict[str, Any],
        signal_windows_list: List[Dict[str, Dict[str, str]]],
        rule_meta: Dict[str, Any],
        retrieval_by_query: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """构建最终分析提示：按你的三问输出最终结论。"""
        # 安全处理可能为 None 的参数
        direction = direction or 'general_analysis'
        context = context or ''
        stats = stats or {}
        signals = signals or []
        validation = validation or {}
        refine_result = refine_result or {}
        signal_windows_list = signal_windows_list or []
        rule_meta = rule_meta or {}
        retrieval_by_query = retrieval_by_query or {}
        
        signal_stats = stats.get('signal_stats', {}) if isinstance(stats, dict) else {}
        
        # 构建信号统计描述
        signal_descriptions = []
        for signal_name in signals[:10]:  # 限制显示前10个信号
            if signal_name in signal_stats:
                signal_data = signal_stats[signal_name]
                if isinstance(signal_data, dict) and signal_data.get('type') == 'numeric':
                    signal_descriptions.append(
                        f"  - {signal_name}: 均值={signal_data.get('mean', 0):.2f}, "
                        f"范围=[{signal_data.get('min', 0):.2f}, {signal_data.get('max', 0):.2f}]"
                    )
                elif isinstance(signal_data, dict) and signal_data.get('type') == 'categorical':
                    dist = signal_data.get('distribution', {})
                    if dist:
                        signal_descriptions.append(
                            f"  - {signal_name}: 分布={dist}"
                        )
        
        signals_text = "\n".join(signal_descriptions) if signal_descriptions else "  - 无详细统计"
        
        # 最终输出严格要求：补充三问结构
        # 1) 是否 2015 标准桩（基于 CHM_ComVersion / 检索证据）
        # 2) 流程问题：车/桩/阶段定位
        # 3) 原因与建议
        payload = {
            "direction": direction,
            "signal_windows": signal_windows_list,
            "signal_rule_meta": rule_meta,
            "rag_evidence": retrieval_by_query,
            "refine_result": refine_result,
            "data_stats": {
                "total_records": stats.get("total_records", 0) if isinstance(stats, dict) else 0,
                "signal_count": stats.get("signal_count", 0) if isinstance(stats, dict) else 0,
                "time_range": stats.get("time_range", {}) if isinstance(stats, dict) else {},
                "key_signal_stats": signal_descriptions[:10],
            },
            "validation": validation,
            "fallback_context": context[:1500],
            "requirements": {
                "output_json_schema": {
                    "is_gbt2015_pile": "boolean|null",
                    "pile_protocol": "string",
                    "process_assessment": {
                        "has_process_issue": "boolean",
                        "responsibility": "car|pile|unknown|both",
                        "stage_breakdown": [{"stage": "string", "issue": "string", "evidence": ["string", "..."]}],
                    },
                    "root_causes": ["string", "..."],
                    "recommendations": ["string", "..."],
                    "summary": "string",
                    "confidence": "number(0~1)",
                }
            },
        }

        prompt = (
            "请基于输入的关键信号时间段变化、枚举释义与知识库证据，回答并只返回严格 JSON：\n"
            "1) 是否是标准2015充电桩？\n"
            "2) 整个充电流程有问题吗？是车的问题还是桩的问题？哪个阶段有什么问题？\n"
            "3) 可能的原因及优化建议？\n"
            "注意：若证据不足可返回 unknown/null，但要降低 confidence。\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        return prompt
    
    async def _llm_analysis(self, prompt: str) -> Dict[str, Any]:
        """LLM分析"""
        # 如果配置了真实的 LLM 客户端，则调用真实 API
        if self.llm_client and self.model_name:
            try:
                logger.info(f"调用 LLM API: model={self.model_name}")
                
                # 构建消息，要求返回 JSON 格式
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个专业的充电系统分析专家。请根据提供的充电数据分析结果，生成详细的诊断报告。必须以有效的 JSON 格式返回结果，包含以下字段：summary（诊断总结）、findings（关键发现列表）、risk_assessment（风险评估）、recommendations（建议措施列表）、technical_details（技术细节）。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                # 使用 stream，避免网关长连接读超时（504）
                stream = await self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    stream=True,
                )

                chunks: list[str] = []
                async for event in stream:
                    try:
                        delta = event.choices[0].delta
                        if delta and delta.content:
                            chunks.append(delta.content)
                    except Exception:
                        continue

                content = "".join(chunks)
                logger.debug(f"LLM 响应: {content[:200]}...")
                
                # 尝试解析 JSON 响应
                try:
                    import json
                    # 尝试提取 JSON 部分（如果响应包含其他文本）
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        analysis_result = json.loads(json_str)
                    else:
                        # 如果整个内容就是 JSON
                        analysis_result = json.loads(content)
                    
                    # 验证必需字段
                    if not isinstance(analysis_result, dict):
                        raise ValueError("LLM 返回的不是有效的 JSON 对象")
                    
                    # 确保所有必需字段存在
                    result = {
                        "summary": analysis_result.get("summary", "分析完成"),
                        "findings": analysis_result.get("findings", []),
                        "risk_assessment": analysis_result.get("risk_assessment", "未知"),
                        "recommendations": analysis_result.get("recommendations", []),
                        "technical_details": analysis_result.get("technical_details", "")
                    }
                    
                    logger.info("LLM 分析完成")
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"LLM 返回的 JSON 解析失败: {e}，使用文本作为摘要")
                    # 如果 JSON 解析失败，将内容作为摘要返回
                    return {
                        "summary": content[:500],
                        "findings": [],
                        "risk_assessment": "未知",
                        "recommendations": [],
                        "technical_details": content
                    }
                    
            except Exception as e:
                # 尽可能打印 HTTP 状态/响应体片段，便于定位 504 来自哪层网关
                try:
                    from openai import APIStatusError
                    if isinstance(e, APIStatusError):
                        body = ""
                        try:
                            body = (e.response.text or "")[:800]
                        except Exception:
                            body = ""
                        logger.error(
                            "调用 LLM API 失败 APIStatusError status=%s body=%s",
                            getattr(e, "status_code", None),
                            body,
                            exc_info=True,
                        )
                    else:
                        logger.error(f"调用 LLM API 失败: {e}，回退到模拟分析", exc_info=True)
                except Exception:
                    logger.error(f"调用 LLM API 失败: {e}，回退到模拟分析", exc_info=True)
                # 发生错误时回退到模拟分析
                return await self._mock_llm_analysis()
        else:
            # 使用模拟分析
            return await self._mock_llm_analysis()
    
    async def _mock_llm_analysis(self) -> Dict[str, Any]:
        """模拟 LLM 分析（当未配置真实 API 时使用）"""
        await asyncio.sleep(1.5)  # 模拟分析时间
        
        mock_analysis = {
            "summary": "检测到充电系统电流异常波动，可能影响充电效率。",
            "findings": [
                "充电过程中出现3次电流突降",
                "电池SOC显示正常充电曲线",
                "BMS状态切换频繁"
            ],
            "risk_assessment": "中等",
            "recommendations": [
                "检查充电连接器接触情况",
                "监控温度传感器数据",
                "优化充电策略参数"
            ],
            "technical_details": "基于信号分析，系统在充电阶段表现出不稳定的电流特征，建议进行硬件检查。"
        }
        
        return mock_analysis


class ReportGenerationNode:
    """报告生成节点"""
    
    def __init__(self):
        self.template_engine = None  # 实际使用时需要初始化模板引擎
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """生成最终报告"""
        analysis_data = {
            'flow_analysis': state.get('flow_analysis'),
            'retrieved_documents': state.get('retrieved_documents', []),
            'llm_analysis': state.get('llm_analysis'),
            'data_stats': state.get('data_stats'),
            'refined_signals': state.get('refined_signals', []),
            'validation': state.get('signal_validation')
        }
        mark_node_started(state, "report_generation", "生成可视化与报告")
        
        try:
            # 生成报告
            report = await self._generate_report(analysis_data)
            
            # 生成可视化数据
            visualizations = await self._generate_visualizations(state)
            
            result_state = {
                **state,
                'final_report': report,
                'visualizations': visualizations,
                'analysis_status': AnalysisStatus.COMPLETED,
                'end_time': datetime.now()
            }
            mark_node_completed(result_state, "report_generation", {
                "visualization_count": len(visualizations),
                "has_summary": bool(report.get('summary'))
            })
            return result_state
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            failed_state = {
                **state,
                'final_report': {
                    'html_content': '报告生成失败',
                    'summary': '分析过程中发生错误',
                    'detailed_analysis': state.get('llm_analysis', {}),
                    'metadata': {'generated_at': datetime.now().isoformat()}
                },
                'error_message': str(e),
                'analysis_status': AnalysisStatus.FAILED
            }
            mark_node_failed(failed_state, "report_generation", str(e))
            return failed_state
    
    async def _generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告"""
        # 模拟报告生成
        await asyncio.sleep(0.5)
        
        analysis = data.get('llm_analysis') or {}
        summary = self._generate_summary(data)
        
        # 安全获取 confidence_score
        flow_analysis = data.get('flow_analysis')
        confidence_score = 0.0
        if flow_analysis and isinstance(flow_analysis, dict):
            confidence_score = flow_analysis.get('confidence', 0.0)
        
        return {
            # 简化版 HTML：把三问的关键字段展示出来（前端如有独立渲染可忽略此 html_content）
            'html_content': (
                "<html><body>"
                "<h1>充电分析报告</h1>"
                f"<p>{summary}</p>"
                f"<h2>2015国标桩判断</h2><pre>{json.dumps({'is_gbt2015_pile': analysis.get('is_gbt2015_pile'), 'pile_protocol': analysis.get('pile_protocol')}, ensure_ascii=False)}</pre>"
                f"<h2>流程评估</h2><pre>{json.dumps(analysis.get('process_assessment') or {}, ensure_ascii=False)}</pre>"
                f"<h2>可能原因</h2><pre>{json.dumps(analysis.get('root_causes') or [], ensure_ascii=False)}</pre>"
                f"<h2>优化建议</h2><pre>{json.dumps(analysis.get('recommendations') or [], ensure_ascii=False)}</pre>"
                "</body></html>"
            ),
            'summary': summary,
            'detailed_analysis': analysis,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0',
                'confidence_score': confidence_score
            }
        }
    
    def _generate_summary(self, data: Dict[str, Any]) -> str:
        """生成摘要"""
        analysis = data.get('llm_analysis') or {}
        validation = data.get('validation') or {}
        
        summary_parts = []
        if analysis and isinstance(analysis, dict):
            if 'summary' in analysis and analysis.get("summary"):
                summary_parts.append(f"诊断结论：{analysis['summary']}")
            if analysis.get("confidence") is not None:
                try:
                    summary_parts.append(f"置信度：{float(analysis.get('confidence')):.0%}")
                except Exception:
                    pass
            if analysis.get("pile_protocol"):
                summary_parts.append(f"协议/桩类型：{analysis.get('pile_protocol')}")
        
        if validation and isinstance(validation, dict) and validation.get('validated'):
            signal_count = validation.get('signal_count', 0)
            summary_parts.append(f"基于{signal_count}个高质量信号进行分析")
        
        if analysis and isinstance(analysis, dict) and analysis.get("process_assessment"):
            pa = analysis.get("process_assessment") or {}
            if isinstance(pa, dict) and "responsibility" in pa:
                summary_parts.append(f"责任判定：{pa.get('responsibility')}")
        
        if not summary_parts:
            return "分析完成，未发现明显异常。"
        
        return "；".join(summary_parts)
    
    async def _generate_visualizations(self, state: ChargingAnalysisState) -> List[Dict[str, Any]]:
        """生成可视化数据"""
        df = state.get('parsed_data')
        if df is None:
            return []
        
        visualizations = []
        
        # 排除元数据列
        metadata_columns = {'timestamp', 'ts', 'time', 'Time', 'can_id', 'message_name', 'dlc'}
        signal_columns = [col for col in df.columns if col not in metadata_columns]
        
        if not signal_columns:
            return visualizations
        
        # 时间序列图 - 使用前几个数值型信号
        numeric_signals = [col for col in signal_columns 
                          if pd.api.types.is_numeric_dtype(df[col])][:3]
        if numeric_signals:
            visualizations.append({
                'type': 'line_chart',
                'title': '信号变化趋势',
                'signals': numeric_signals
            })
        
        # 分布直方图 - 使用第一个数值型信号
        if numeric_signals:
            visualizations.append({
                'type': 'histogram',
                'title': f'{numeric_signals[0]} 分布',
                'signals': [numeric_signals[0]]
            })
        
        # 分类信号分布图
        categorical_signals = [col for col in signal_columns 
                              if not pd.api.types.is_numeric_dtype(df[col]) 
                              and df[col].nunique() <= 10][:2]
        if categorical_signals:
            visualizations.append({
                'type': 'bar_chart',
                'title': '状态分布',
                'signals': categorical_signals
            })
        
        return visualizations


class ChargingAnalysisWorkflow:
    """充电分析工作流主类"""
    
    def __init__(self):
        self.graph = StateGraph(ChargingAnalysisState)
        self._build_workflow()
        self._app = self.graph.compile()
    
    def _build_workflow(self):
        """构建工作流图"""
        # 创建节点实例
        file_validation = FileValidationNode()
        message_parsing = MessageParsingNode()
        flow_control = FlowControlModelNode()
        rag_retrieval = RAGRetrievalNode()
        detailed_analysis = DetailedAnalysisNode()
        llm_analysis = LLMAnalysisNode()
        report_generation = ReportGenerationNode()
        
        # 添加节点
        self.graph.add_node("file_validation", file_validation.process)
        self.graph.add_node("message_parsing", message_parsing.process)
        self.graph.add_node("flow_control", flow_control.process)
        self.graph.add_node("rag_retrieval", rag_retrieval.process)
        self.graph.add_node("detailed_analysis", detailed_analysis.process)
        self.graph.add_node("llm_analysis", llm_analysis.process)
        self.graph.add_node("report_generation", report_generation.process)
        
        # 添加边
        self.graph.add_edge(START, "file_validation")
        self.graph.add_edge("file_validation", "message_parsing")
        self.graph.add_edge("message_parsing", "flow_control")
        self.graph.add_edge("flow_control", "rag_retrieval")
        self.graph.add_edge("rag_retrieval", "detailed_analysis")
        # 细化分析后根据置信度决定：回到 flow_control 继续补信号，或进入最终 LLM 分析
        self.graph.add_conditional_edges(
            "detailed_analysis",
            self.should_continue_analysis,
            {
                "flow_control": "flow_control",
                "llm_analysis": "llm_analysis",
            },
        )
        self.graph.add_edge("llm_analysis", "report_generation")
        self.graph.add_edge("report_generation", END)
    
    async def execute(self, initial_state: ChargingAnalysisState) -> ChargingAnalysisState:
        """执行工作流"""
        try:
            # 验证初始状态
            required_fields = ['analysis_id', 'file_path', 'file_name', 'file_size', 'user_id']
            for field in required_fields:
                if field not in initial_state or initial_state[field] is None:
                    raise ValueError(f"缺少必要字段: {field}")
            
            # 添加默认字段
            initial_state.setdefault('iteration', 0)
            initial_state.setdefault('start_time', datetime.now())
            initial_state.setdefault('analysis_status', AnalysisStatus.PENDING)
            initial_state.setdefault('workflow_trace', {})
            initial_state.setdefault('parsed_data_records', None)
            initial_state.setdefault('selected_signals', None)
            
            logger.info(f"开始执行充电分析工作流: {initial_state['analysis_id']}")
            logger.info(f"initial_state 中的 selected_signals: {initial_state.get('selected_signals')}")
            
            # 执行工作流
            final_state = await self._app.ainvoke(initial_state)
            
            logger.info(f"工作流执行完成: {final_state['analysis_id']}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            
            # 返回错误状态
            return {
                **initial_state,
                'analysis_status': AnalysisStatus.FAILED,
                'error_message': str(e),
                'end_time': datetime.now()
            }
    
    def should_continue_analysis(self, state: ChargingAnalysisState) -> str:
        """细化分析后决定：是否回到流程控制补信号。"""
        iteration = int(state.get("iteration") or 0)
        conf = float(state.get("refine_confidence") or 0.0)
        additional = state.get("additional_signals") or []

        # 达到阈值：进入最终 LLM 分析
        if conf >= 0.7:
            return "llm_analysis"

        # 超过最大轮次：也进入最终 LLM（但置信度低，结论需要提示不确定）
        if iteration >= 3:
            return "llm_analysis"

        # 有可补充信号：回到流程控制继续
        if additional:
            return "flow_control"

        return "llm_analysis"