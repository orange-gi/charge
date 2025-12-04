"""
LangGraph 工作流实现

包含所有 Agent 节点定义、工作流编排和状态管理。
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from enum import Enum

import pandas as pd
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import Checkpointer

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
    
    # RAG检索
    retrieved_documents: Optional[List[Dict[str, Any]]]
    retrieval_context: Optional[str]
    retrieval_status: Optional[str]
    
    # 细化分析
    iteration: int
    refined_signals: Optional[List[str]]
    signal_validation: Optional[Dict[str, Any]]
    analysis_status: AnalysisStatus
    
    # 最终结果
    llm_analysis: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    visualizations: Optional[List[Dict[str, Any]]]
    
    # 元数据
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str]
    progress_callback: Optional[callable]


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
            
            return {
                **state,
                'validation_status': 'passed',
                'validation_message': '文件验证通过'
            }
            
        except Exception as e:
            logger.error(f"文件验证失败: {str(e)}")
            if state.get('progress_callback'):
                await state['progress_callback']("文件验证", 0, f"文件验证失败: {str(e)}")
            
            return {
                **state,
                'validation_status': 'failed',
                'error_message': str(e),
                'analysis_status': AnalysisStatus.FAILED
            }
    
    async def _check_file_integrity(self, file_path: str) -> bool:
        """检查文件完整性"""
        try:
            import os
            return os.path.exists(file_path) and os.path.getsize(file_path) > 0
        except Exception:
            return False


class MessageParsingNode:
    """报文解析节点"""
    
    def __init__(self):
        self.filter_signals = [
            'CHM_ComVersion', 'BMS_DCChrgSt', 'CCS_OutputCurent',
            'CRM_RecognitionResult', 'BMS_DCChrgConnectSt',
            'BMS_BattCurrt', 'BCL_CurrentRequire', 'BMS_ChrgEndNum',
            'BMS_PackSOCDisp', 'CRO_ChargeReady', 'BRO_BMSChargeReady'
        ]
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理报文解析"""
        file_path = state['file_path']
        
        try:
            # 模拟解析过程
            if state.get('progress_callback'):
                await state['progress_callback']("报文解析", 20, "开始解析报文数据...")
            
            # 实际解析逻辑（这里用模拟数据）
            df_parsed = await self._parse_file(file_path, self.filter_signals)
            
            # 数据预处理和清洗
            df_cleaned = await self._clean_data(df_parsed)
            
            # 提取关键统计信息
            stats = await self._extract_statistics(df_cleaned)
            
            if state.get('progress_callback'):
                await state['progress_callback']("报文解析", 40, "报文解析完成")
            
            return {
                **state,
                'parsed_data': df_cleaned,
                'data_stats': stats,
                'parsing_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"报文解析失败: {str(e)}")
            if state.get('progress_callback'):
                await state['progress_callback']("报文解析", 0, f"报文解析失败: {str(e)}")
            
            return {
                **state,
                'parsing_status': 'failed',
                'error_message': str(e),
                'analysis_status': AnalysisStatus.FAILED
            }
    
    async def _parse_file(self, file_path: str, filter_signals: List[str]) -> pd.DataFrame:
        """解析文件"""
        # 模拟解析过程
        await asyncio.sleep(2)  # 模拟处理时间
        
        # 生成模拟数据
        import numpy as np
        n_samples = 1000
        
        data = {
            'ts': pd.date_range('2025-11-19', periods=n_samples, freq='1S'),
            'BMS_DCChrgSt': np.random.choice([0, 1, 2, 3], n_samples),
            'BMS_BattCurrt': np.random.normal(25, 5, n_samples),
            'BCL_CurrentRequire': np.random.normal(30, 3, n_samples),
            'BMS_PackSOCDisp': np.random.uniform(20, 90, n_samples),
            'BMS_ChrgEndNum': np.random.randint(0, 10, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['ts'] = pd.to_datetime(df['ts'])
        
        return df
    
    async def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 移除空值
        df_cleaned = df.dropna(subset=['BMS_DCChrgSt'])
        
        # 移除异常值
        df_cleaned = df_cleaned[
            (df_cleaned['BMS_BattCurrt'] > -100) & 
            (df_cleaned['BMS_BattCurrt'] < 100)
        ]
        
        return df_cleaned
    
    async def _extract_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """提取统计信息"""
        stats = {
            'total_records': len(df),
            'time_range': {
                'start': df['ts'].min().isoformat(),
                'end': df['ts'].max().isoformat()
            },
            'signal_stats': {
                'BMS_DCChrgSt': {
                    'unique_values': df['BMS_DCChrgSt'].unique().tolist(),
                    'distribution': df['BMS_DCChrgSt'].value_counts().to_dict()
                },
                'BMS_BattCurrt': {
                    'mean': float(df['BMS_BattCurrt'].mean()),
                    'std': float(df['BMS_BattCurrt'].std()),
                    'min': float(df['BMS_BattCurrt'].min()),
                    'max': float(df['BMS_BattCurrt'].max())
                }
            }
        }
        
        return stats


class FlowControlModelNode:
    """流程控制模型节点"""
    
    def __init__(self):
        self.signal_definitions = {
            'charging_status': 'BMS_DCChrgSt',
            'battery_current': 'BMS_BattCurrt', 
            'current_requirement': 'BCL_CurrentRequire'
        }
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理流程控制模型分析"""
        df = state.get('parsed_data')
        stats = state.get('data_stats')
        
        if df is None or stats is None:
            raise ValueError("缺少解析数据")
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("流程分析", 50, "流程控制模型分析中...")
            
            # 构建输入提示
            prompt = self._build_prompt(stats)
            
            # 模拟模型推理
            result = await self._model_inference(prompt)
            
            # 解析结果
            analysis_result = self._parse_model_output(result)
            
            if state.get('progress_callback'):
                await state['progress_callback']("流程分析", 60, "流程分析完成")
            
            return {
                **state,
                'flow_analysis': analysis_result,
                'problem_direction': analysis_result.get('problem_direction', 'general_analysis'),
                'confidence_score': analysis_result.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"流程控制模型分析失败: {str(e)}")
            return {
                **state,
                'problem_direction': 'general_analysis',
                'confidence_score': 0.5,
                'error_message': str(e)
            }
    
    def _build_prompt(self, stats: Dict[str, Any]) -> str:
        """构建提示"""
        prompt = f"""
充电数据分析：
总记录数: {stats['total_records']}
时间范围: {stats['time_range']['start']} 至 {stats['time_range']['end']}

关键信号统计：
- 充电状态分布: {stats['signal_stats']['BMS_DCChrgSt']['distribution']}
- 电池电流统计: 均值={stats['signal_stats']['BMS_BattCurrt']['mean']:.2f}, 标准差={stats['signal_stats']['BMS_BattCurrt']['std']:.2f}

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
        self.vector_store = None  # 实际使用时需要初始化ChromaDB客户端
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理RAG检索"""
        problem_direction = state.get('problem_direction', '')
        df = state.get('parsed_data')
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("知识检索", 70, "RAG知识检索中...")
            
            # 构建检索查询
            query = self._build_retrieval_query(problem_direction, df)
            
            # 检索相关文档
            documents = await self._retrieve_documents(query)
            
            # 构建上下文
            context = self._build_context(documents)
            
            if state.get('progress_callback'):
                await state['progress_callback']("知识检索", 80, "知识检索完成")
            
            return {
                **state,
                'retrieved_documents': documents,
                'retrieval_context': context,
                'retrieval_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"RAG检索失败: {str(e)}")
            return {
                **state,
                'retrieval_status': 'failed',
                'error_message': str(e)
            }
    
    def _build_retrieval_query(self, problem_direction: str, df: pd.DataFrame) -> str:
        """构建检索查询"""
        base_query = f"充电系统{problem_direction}"
        
        # 基于数据特征调整查询
        if 'BMS_DCChrgSt' in df.columns:
            status_dist = df['BMS_DCChrgSt'].value_counts().to_dict()
            if 0 in status_dist:
                base_query += " 充电连接问题"
            if 2 in status_dist:
                base_query += " 充电中异常"
        
        return base_query
    
    async def _retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索文档"""
        # 模拟检索过程
        await asyncio.sleep(0.5)
        
        # 返回模拟文档
        mock_documents = [
            {
                'content': f'关于{query}的技术文档内容，包括故障诊断和处理方法。',
                'metadata': {'source': '技术手册.pdf', 'category': 'troubleshooting'},
                'score': 0.89
            },
            {
                'content': f'{query}相关的最佳实践和解决方案。',
                'metadata': {'source': '操作指南.docx', 'category': 'guide'},
                'score': 0.76
            }
        ]
        
        return mock_documents
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """构建上下文"""
        context_parts = []
        for doc in documents:
            context_parts.append(f"[{doc['score']:.3f}] {doc['content']}")
        return "\n\n".join(context_parts)


class DetailedAnalysisNode:
    """细化分析节点"""
    
    def __init__(self):
        self.max_iterations = 3
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理细化分析"""
        problem_direction = state.get('problem_direction', '')
        context = state.get('retrieval_context', '')
        df = state.get('parsed_data')
        iteration = state.get('iteration', 0)
        
        if iteration >= self.max_iterations:
            if state.get('progress_callback'):
                await state['progress_callback']("细化分析", 85, "达到最大迭代次数")
            
            return {
                **state,
                'analysis_status': AnalysisStatus.MAX_ITERATIONS,
                'iteration': iteration
            }
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("细化分析", 85, f"细化分析第{iteration+1}次")
            
            # 提取细化信号
            refined_signals = await self._extract_refined_signals(
                problem_direction, context, df
            )
            
            # 验证信号
            signal_validation = await self._validate_signals(df, refined_signals)
            
            # 如果验证失败，尝试下一个迭代
            if not signal_validation['validated']:
                return {
                    **state,
                    'iteration': iteration + 1,
                    'refined_signals': refined_signals,
                    'signal_validation': signal_validation,
                    'next_step': 'flow_control_retry'
                }
            
            if state.get('progress_callback'):
                await state['progress_callback']("细化分析", 90, "细化分析验证通过")
            
            return {
                **state,
                'refined_signals': refined_signals,
                'signal_validation': signal_validation,
                'analysis_status': AnalysisStatus.COMPLETED
            }
            
        except Exception as e:
            logger.error(f"细化分析失败: {str(e)}")
            return {
                **state,
                'analysis_status': AnalysisStatus.VALIDATION_FAILED,
                'error_message': str(e)
            }
    
    async def _extract_refined_signals(self, direction: str, context: str, df: pd.DataFrame) -> List[str]:
        """提取细化信号"""
        signal_mapping = {
            'charging_connection': ['BMS_DCChrgConnectSt', 'BMS_Cc2SngR'],
            'charging_current': ['BCL_CurrentRequire', 'BMS_ChrgCurrtLkUp'],
            'charging_voltage': ['CML_OutputVoltageMax', 'CML_OutputCurentMax'],
            'battery_soc': ['BMS_PackSOCDisp']
        }
        
        relevant_signals = []
        for category, signals in signal_mapping.items():
            if category in direction:
                relevant_signals.extend(signals)
        
        # 确保信号在数据中存在
        available_signals = [s for s in relevant_signals if s in df.columns]
        
        return list(set(available_signals))
    
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
        self.llm_client = None  # 实际使用时需要初始化LLM客户端
    
    async def process(self, state: ChargingAnalysisState) -> ChargingAnalysisState:
        """处理LLM分析"""
        problem_direction = state.get('problem_direction')
        context = state.get('retrieval_context', '')
        data_stats = state.get('data_stats', {})
        refined_signals = state.get('refined_signals', [])
        validation = state.get('signal_validation', {})
        
        try:
            if state.get('progress_callback'):
                await state['progress_callback']("LLM分析", 95, "LLM生成分析报告...")
            
            # 构建分析提示
            prompt = self._build_analysis_prompt(
                problem_direction, context, data_stats, refined_signals, validation
            )
            
            # 模拟LLM分析
            analysis_result = await self._llm_analysis(prompt)
            
            if state.get('progress_callback'):
                await state['progress_callback']("LLM分析", 100, "分析报告生成完成")
            
            return {
                **state,
                'llm_analysis': analysis_result,
                'analysis_status': AnalysisStatus.COMPLETED
            }
            
        except Exception as e:
            logger.error(f"LLM分析失败: {str(e)}")
            return {
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
    
    def _build_analysis_prompt(self, direction: str, context: str, stats: Dict, 
                              signals: List[str], validation: Dict) -> str:
        """构建分析提示"""
        prompt = f"""
基于以下充电数据分析，生成详细的诊断报告：

问题方向：{direction}
相关信号：{', '.join(signals)}
信号验证结果：{validation.get('average_quality', 0):.2f}

数据统计：
- 总记录数：{stats.get('total_records', 0)}
- 时间范围：{stats.get('time_range', {})}
- 充电状态分布：{stats.get('signal_stats', {}).get('BMS_DCChrgSt', {}).get('distribution', {})}

相关知识：
{context[:1000]}...

请生成包含以下内容的诊断报告：
1. 问题诊断总结
2. 关键发现
3. 风险评估
4. 建议措施
5. 技术细节

以JSON格式返回结果。
"""
        return prompt
    
    async def _llm_analysis(self, prompt: str) -> Dict[str, Any]:
        """LLM分析"""
        await asyncio.sleep(1.5)  # 模拟分析时间
        
        # 模拟分析结果
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
        
        try:
            # 生成报告
            report = await self._generate_report(analysis_data)
            
            # 生成可视化数据
            visualizations = await self._generate_visualizations(state)
            
            return {
                **state,
                'final_report': report,
                'visualizations': visualizations,
                'analysis_status': AnalysisStatus.COMPLETED,
                'end_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            return {
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
    
    async def _generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告"""
        # 模拟报告生成
        await asyncio.sleep(0.5)
        
        analysis = data.get('llm_analysis', {})
        summary = self._generate_summary(data)
        
        return {
            'html_content': f"<html><body><h1>充电分析报告</h1><p>{summary}</p></body></html>",
            'summary': summary,
            'detailed_analysis': analysis,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0',
                'confidence_score': data.get('flow_analysis', {}).get('confidence', 0)
            }
        }
    
    def _generate_summary(self, data: Dict[str, Any]) -> str:
        """生成摘要"""
        analysis = data.get('llm_analysis', {})
        validation = data.get('validation', {})
        
        summary_parts = []
        if 'summary' in analysis:
            summary_parts.append(f"诊断结论：{analysis['summary']}")
        
        if validation.get('validated'):
            signal_count = validation.get('signal_count', 0)
            summary_parts.append(f"基于{signal_count}个高质量信号进行分析")
        
        if 'risk_assessment' in analysis:
            risk = analysis['risk_assessment']
            summary_parts.append(f"风险评估：{risk}")
        
        return "；".join(summary_parts)
    
    async def _generate_visualizations(self, state: ChargingAnalysisState) -> List[Dict[str, Any]]:
        """生成可视化数据"""
        df = state.get('parsed_data')
        if df is None:
            return []
        
        visualizations = []
        
        # 时间序列图
        if 'BMS_DCChrgSt' in df.columns:
            visualizations.append({
                'type': 'line_chart',
                'title': '充电状态变化趋势',
                'signals': ['BMS_DCChrgSt', 'BMS_BattCurrt']
            })
        
        # 电流分布直方图
        if 'BMS_BattCurrt' in df.columns:
            visualizations.append({
                'type': 'histogram',
                'title': '电池电流分布'
            })
        
        return visualizations


class ChargingAnalysisWorkflow:
    """充电分析工作流主类"""
    
    def __init__(self):
        self.graph = StateGraph(ChargingAnalysisState)
        self._build_workflow()
    
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
        self.graph.add_edge("__root__", "file_validation")
        self.graph.add_edge("file_validation", "message_parsing")
        self.graph.add_edge("message_parsing", "flow_control")
        self.graph.add_edge("flow_control", "rag_retrieval")
        self.graph.add_edge("rag_retrieval", "detailed_analysis")
        self.graph.add_edge("detailed_analysis", "llm_analysis")
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
            
            logger.info(f"开始执行充电分析工作流: {initial_state['analysis_id']}")
            
            # 执行工作流
            final_state = await self.graph.ainvoke(initial_state)
            
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
        """决定是否继续分析"""
        validation = state.get('signal_validation', {})
        
        # 如果验证通过，进行LLM分析
        if validation.get('validated'):
            return "llm_analysis"
        
        # 如果达到最大迭代次数，结束分析
        iteration = state.get('iteration', 0)
        if iteration >= 3:
            return "end"
        
        # 否则继续下一轮分析
        return "continue"