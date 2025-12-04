# 后端 LangGraph 架构设计

## 1. LangGraph 概述

LangGraph 是基于 LangChain 构建的 Agent 编排框架，支持复杂的多 Agent 工作流编排、状态管理和流程控制。本系统使用 LangGraph 来构建充电分析、RAG 检索和训练管理等核心业务流程。

### 1.1 核心特性
- **图形化工作流**: 基于图结构的 Agent 编排
- **状态管理**: 内置状态传递和持久化
- **条件分支**: 支持复杂的流程控制逻辑
- **错误处理**: 完整的异常处理和重试机制
- **监控和调试**: 工作流执行监控和调试工具

## 2. 系统工作流架构

### 2.1 整体工作流图

```
用户请求 → API网关 → 认证中间件 → 工作流路由器 → LangGraph执行器
                                     ↓
配置管理 ← 状态存储 ← 结果缓存 ← 工作流执行引擎 ← 外部服务调用
```

### 2.2 核心工作流分类

1. **充电分析工作流**: 数据上传 → 信号处理 → 问题诊断 → RAG检索 → 结果生成
2. **RAG检索工作流**: 查询解析 → 向量检索 → 内容重排 → 上下文构建
3. **训练管理工作流**: 任务创建 → 数据验证 → 训练执行 → 模型评估 → 版本管理
4. **用户管理工作流**: 认证 → 授权 → 权限检查 → 操作审计

## 3. 充电分析工作流详细设计

### 3.1 工作流图

```
[开始] → [文件验证] → [报文解析] → [信号提取] → [流程控制模型]
    ↓                                                   ↓
[结束] ← [生成报告] ← [LLM分析] ← [RAG检索] ← [问题方向确定]
                                      ↓
                                [信号验证] ← [细化分析]
```

### 3.2 节点定义

#### 3.2.1 文件验证节点 (FileValidationNode)
```python
from langgraph import Node
from typing import Dict, Any

class FileValidationNode(Node):
    def __init__(self):
        self.allowed_extensions = ['.blf', '.csv', '.xlsx']
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        file_path = state.get('file_path')
        file_size = state.get('file_size')
        file_name = state.get('file_name')
        
        # 验证文件类型
        if not any(file_name.endswith(ext) for ext in self.allowed_extensions):
            raise ValueError(f"不支持的文件类型: {file_name}")
        
        # 验证文件大小
        if file_size > self.max_file_size:
            raise ValueError(f"文件过大: {file_size / (1024*1024):.1f}MB")
        
        # 验证文件完整性
        if not await self._check_file_integrity(file_path):
            raise ValueError("文件损坏或格式错误")
        
        return {
            **state,
            'validation_status': 'passed',
            'validation_message': '文件验证通过'
        }
    
    async def _check_file_integrity(self, file_path: str) -> bool:
        # 实现文件完整性检查
        pass
```

#### 3.2.2 报文解析节点 (MessageParsingNode)
```python
class MessageParsingNode(Node):
    def __init__(self):
        self.parser = CanParse()
        self.filter_signals = [
            'CHM_ComVersion', 'BMS_DCChrgSt', 'CCS_OutputCurent',
            'CRM_RecognitionResult', 'BMS_DCChrgConnectSt',
            'BMS_BattCurrt', 'BCL_CurrentRequire', 'BMS_ChrgEndNum'
        ]
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        file_path = state['file_path']
        
        # 解析报文数据
        df_parsed = await self.parser.parse_file(
            file_path, 
            self.filter_signals,
            state.get('progress_callback')
        )
        
        # 数据预处理和清洗
        df_cleaned = await self._clean_data(df_parsed)
        
        # 提取关键统计信息
        stats = await self._extract_statistics(df_cleaned)
        
        return {
            **state,
            'parsed_data': df_cleaned,
            'data_stats': stats,
            'parsing_status': 'completed'
        }
    
    async def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 数据清洗逻辑
        df_cleaned = df.dropna(subset=['BMS_DCChrgSt'])
        df_cleaned = df_cleaned[(df_cleaned['BMS_BattCurrt'] > -100) & 
                               (df_cleaned['BMS_BattCurrt'] < 100)]
        return df_cleaned
    
    async def _extract_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
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
```

#### 3.2.3 流程控制模型节点 (FlowControlModelNode)
```python
class FlowControlModelNode(Node):
    def __init__(self):
        self.model = self._load_model()
        self.signal_definitions = {
            'charging_status': 'BMS_DCChrgSt',
            'battery_current': 'BMS_BattCurrt', 
            'current_requirement': 'BCL_CurrentRequire'
        }
    
    def _load_model(self):
        # 加载1.5B流程控制模型
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = config.SMALL_MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return {
            'model': model,
            'tokenizer': tokenizer
        }
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df = state['parsed_data']
        stats = state['data_stats']
        
        # 构建输入提示
        prompt = self._build_prompt(stats)
        
        # 模型推理
        result = await self._model_inference(prompt)
        
        # 解析结果
        analysis_result = self._parse_model_output(result)
        
        return {
            **state,
            'flow_analysis': analysis_result,
            'problem_direction': analysis_result.get('problem_direction'),
            'confidence_score': analysis_result.get('confidence', 0.0)
        }
    
    def _build_prompt(self, stats: Dict[str, Any]) -> str:
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
        tokenizer = self.model['tokenizer']
        model = self.model['model']
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 200,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def _parse_model_output(self, output: str) -> Dict[str, Any]:
        # 解析模型输出
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
```

#### 3.2.4 RAG检索节点 (RAGRetrievalNode)
```python
class RAGRetrievalNode(Node):
    def __init__(self):
        self.vector_store = self._load_vector_store()
        self.reranker = self._load_reranker()
    
    def _load_vector_store(self):
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
        return client
    
    def _load_reranker(self):
        # 加载重排序模型
        from sentence_transformers import CrossEncoder
        return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        problem_direction = state.get('problem_direction', '')
        df = state['parsed_data']
        
        # 构建检索查询
        query = self._build_retrieval_query(problem_direction, df)
        
        # 检索相关文档
        documents = await self._retrieve_documents(query)
        
        # 重排序
        ranked_docs = await self._rerank_documents(query, documents)
        
        # 构建上下文
        context = self._build_context(ranked_docs)
        
        return {
            **state,
            'retrieved_documents': ranked_docs,
            'retrieval_context': context,
            'retrieval_status': 'completed'
        }
    
    def _build_retrieval_query(self, problem_direction: str, df: pd.DataFrame) -> str:
        # 基于问题方向和数据特征构建查询
        base_query = f"充电系统{problem_direction}"
        
        # 添加数据特定信息
        if 'BMS_DCChrgSt' in df.columns:
            status_dist = df['BMS_DCChrgSt'].value_counts().to_dict()
            if 0 in status_dist:
                base_query += " 充电连接问题"
            if 2 in status_dist:
                base_query += " 充电中异常"
        
        return base_query
    
    async def _retrieve_documents(self, query: str, top_k: int = 5):
        collection = self.vector_store.get_collection("charging_knowledge")
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results
    
    async def _rerank_documents(self, query: str, documents: Dict[str, Any]):
        # 重排序文档
        doc_texts = documents['documents'][0]
        reranker_scores = self.reranker.predict([(query, doc) for doc in doc_texts])
        
        # 组合结果
        ranked_results = []
        for i, (doc, score) in enumerate(zip(doc_texts, reranker_scores)):
            ranked_results.append({
                'content': doc,
                'score': float(score),
                'metadata': documents['metadatas'][0][i] if documents['metadatas'] else {}
            })
        
        # 按分数排序
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        return ranked_results
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        context_parts = []
        for doc in documents:
            context_parts.append(f"[{doc['score']:.3f}] {doc['content']}")
        return "\n\n".join(context_parts)
```

#### 3.2.5 细化分析节点 (DetailedAnalysisNode)
```python
class DetailedAnalysisNode(Node):
    def __init__(self):
        self.max_iterations = 3
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        problem_direction = state.get('problem_direction')
        context = state.get('retrieval_context', '')
        df = state['parsed_data']
        iteration = state.get('iteration', 0)
        
        if iteration >= self.max_iterations:
            return {**state, 'analysis_status': 'max_iterations_reached'}
        
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
        
        return {
            **state,
            'refined_signals': refined_signals,
            'signal_validation': signal_validation,
            'analysis_status': 'validated'
        }
    
    async def _extract_refined_signals(self, direction: str, context: str, df: pd.DataFrame) -> List[str]:
        # 基于问题方向和上下文提取相关信号
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
        
        return list(set(relevant_signals))
    
    async def _validate_signals(self, df: pd.DataFrame, signals: List[str]) -> Dict[str, Any]:
        validation_results = []
        for signal in signals:
            if signal in df.columns:
                # 检查信号质量和一致性
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
        # 计算信号质量分数
        non_null_ratio = series.count() / len(series)
        variance = series.var()
        
        # 质量分数 = 非空比例 + 方差合理性
        quality = non_null_ratio * 0.7
        if variance > 0 and variance < 1000:  # 合理方差范围
            quality += 0.3
        elif variance == 0:  # 常数信号，质量较低
            quality += 0.1
        
        return min(quality, 1.0)
```

#### 3.2.6 LLM分析节点 (LLMAnalysisNode)
```python
class LLMAnalysisNode(Node):
    def __init__(self):
        self.llm_client = self._create_llm_client()
    
    def _create_llm_client(self):
        # 创建OpenAI格式的LLM客户端
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        problem_direction = state.get('problem_direction')
        context = state.get('retrieval_context', '')
        data_stats = state.get('data_stats', {})
        refined_signals = state.get('refined_signals', [])
        validation = state.get('signal_validation', {})
        
        # 构建分析提示
        prompt = self._build_analysis_prompt(
            problem_direction, context, data_stats, refined_signals, validation
        )
        
        # 调用LLM进行问题总结
        analysis_result = await self._llm_analysis(prompt)
        
        return {
            **state,
            'llm_analysis': analysis_result,
            'analysis_status': 'completed'
        }
    
    def _build_analysis_prompt(self, direction: str, context: str, stats: Dict, 
                              signals: List[str], validation: Dict) -> str:
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
        response = await self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是电动汽车充电系统诊断专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        try:
            # 尝试解析JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                return result
        except json.JSONDecodeError:
            pass
        
        # 备用格式
        return {
            "summary": content,
            "findings": [],
            "risk_assessment": "中等",
            "recommendations": [],
            "technical_details": ""
        }
```

#### 3.2.7 报告生成节点 (ReportGenerationNode)
```python
class ReportGenerationNode(Node):
    def __init__(self):
        self.template_engine = self._setup_template_engine()
    
    def _setup_template_engine(self):
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader('templates'))
        return env
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        analysis_data = {
            'flow_analysis': state.get('flow_analysis'),
            'retrieved_documents': state.get('retrieved_documents', []),
            'llm_analysis': state.get('llm_analysis'),
            'data_stats': state.get('data_stats'),
            'refined_signals': state.get('refined_signals', []),
            'validation': state.get('signal_validation')
        }
        
        # 生成报告
        report = await self._generate_report(analysis_data)
        
        # 生成可视化数据
        visualizations = await self._generate_visualizations(state)
        
        return {
            **state,
            'final_report': report,
            'visualizations': visualizations,
            'report_status': 'generated'
        }
    
    async def _generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        template = self.template_engine.get_template('charging_analysis_report.html')
        
        report_content = template.render(
            analysis=data['llm_analysis'],
            statistics=data['data_stats'],
            validation=data['validation']
        )
        
        # 生成摘要
        summary = self._generate_summary(data)
        
        return {
            'html_content': report_content,
            'summary': summary,
            'detailed_analysis': data['llm_analysis'],
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0',
                'confidence_score': data.get('flow_analysis', {}).get('confidence', 0)
            }
        }
    
    def _generate_summary(self, data: Dict[str, Any]) -> str:
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
    
    async def _generate_visualizations(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        df = state.get('parsed_data')
        if df is None:
            return []
        
        visualizations = []
        
        # 1. 充电状态时间序列
        if 'BMS_DCChrgSt' in df.columns:
            visualizations.append({
                'type': 'line_chart',
                'title': '充电状态变化趋势',
                'data': self._prepare_time_series_data(df, ['BMS_DCChrgSt', 'BMS_BattCurrt']),
                'signals': ['BMS_DCChrgSt', 'BMS_BattCurrt']
            })
        
        # 2. 电流分布直方图
        if 'BMS_BattCurrt' in df.columns:
            visualizations.append({
                'type': 'histogram',
                'title': '电池电流分布',
                'data': self._prepare_distribution_data(df, 'BMS_BattCurrt')
            })
        
        # 3. 相关性分析
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            visualizations.append({
                'type': 'correlation_matrix',
                'title': '信号相关性分析',
                'data': self._prepare_correlation_data(df[numeric_columns])
            })
        
        return visualizations
    
    def _prepare_time_series_data(self, df: pd.DataFrame, signals: List[str]) -> List[Dict]:
        data = []
        for _, row in df.iterrows():
            data_point = {'timestamp': row['ts'].isoformat()}
            for signal in signals:
                if signal in row and pd.notna(row[signal]):
                    data_point[signal] = float(row[signal])
            data.append(data_point)
        return data
    
    def _prepare_distribution_data(self, df: pd.DataFrame, signal: str) -> List[Dict]:
        values = df[signal].dropna().tolist()
        return [{'value': v} for v in values]
    
    def _prepare_correlation_data(self, df: pd.DataFrame) -> Dict:
        correlation_matrix = df.corr().round(3)
        return {
            'matrix': correlation_matrix.values.tolist(),
            'labels': correlation_matrix.columns.tolist()
        }
```

## 4. 状态管理机制

### 4.1 状态结构定义

```python
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class AnalysisStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATION_FAILED = "validation_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations_reached"

class ChargingAnalysisState(TypedDict):
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
```

### 4.2 状态持久化

```python
class StateManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=0,
            decode_responses=True
        )
    
    async def save_state(self, analysis_id: str, state: ChargingAnalysisState):
        # 序列化状态
        serialized_state = self._serialize_state(state)
        
        # 保存到Redis
        key = f"analysis_state:{analysis_id}"
        await self.redis_client.setex(key, 3600, serialized_state)
    
    async def load_state(self, analysis_id: str) -> Optional[ChargingAnalysisState]:
        key = f"analysis_state:{analysis_id}"
        serialized_state = await self.redis_client.get(key)
        
        if serialized_state:
            return self._deserialize_state(serialized_state)
        return None
    
    async def update_state(self, analysis_id: str, updates: Dict[str, Any]):
        state = await self.load_state(analysis_id)
        if state:
            state.update(updates)
            await self.save_state(analysis_id, state)
    
    def _serialize_state(self, state: ChargingAnalysisState) -> str:
        # 处理不能直接序列化的对象
        serializable_state = {}
        for key, value in state.items():
            if isinstance(value, pd.DataFrame):
                serializable_state[key] = value.to_json()
            elif isinstance(value, datetime):
                serializable_state[key] = value.isoformat()
            else:
                serializable_state[key] = value
        
        return json.dumps(serializable_state, default=str)
    
    def _deserialize_state(self, serialized_state: str) -> ChargingAnalysisState:
        data = json.loads(serialized_state)
        
        # 还原不能直接反序列化的对象
        if 'parsed_data' in data and data['parsed_data']:
            data['parsed_data'] = pd.read_json(data['parsed_data'])
        if 'start_time' in data and data['start_time']:
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        return data
```

## 5. 工作流执行引擎

### 5.1 主工作流定义

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint import Checkpointer

class ChargingAnalysisWorkflow:
    def __init__(self):
        self.graph = StateGraph(ChargingAnalysisState)
        self._build_workflow()
    
    def _build_workflow(self):
        # 添加节点
        self.graph.add_node("file_validation", self.file_validation_node)
        self.graph.add_node("message_parsing", self.message_parsing_node)
        self.graph.add_node("flow_control", self.flow_control_model_node)
        self.graph.add_node("rag_retrieval", self.rag_retrieval_node)
        self.graph.add_node("detailed_analysis", self.detailed_analysis_node)
        self.graph.add_node("llm_analysis", self.llm_analysis_node)
        self.graph.add_node("report_generation", self.report_generation_node)
        self.graph.add_node("error_handler", self.error_handler_node)
        
        # 添加边
        self.graph.add_edge("__root__", "file_validation")
        self.graph.add_edge("file_validation", "message_parsing")
        self.graph.add_edge("message_parsing", "flow_control")
        self.graph.add_edge("flow_control", "rag_retrieval")
        self.graph.add_edge("rag_retrieval", "detailed_analysis")
        
        # 条件边
        self.graph.add_conditional_edges(
            "detailed_analysis",
            self.should_continue_analysis,
            {
                "continue": "flow_control",
                "llm_analysis": "llm_analysis",
                "end": "report_generation"
            }
        )
        
        self.graph.add_edge("llm_analysis", "report_generation")
        self.graph.add_edge("report_generation", "__end__")
        
        # 错误处理边
        self.graph.add_edge("file_validation", "error_handler")
        self.graph.add_edge("message_parsing", "error_handler")
        self.graph.add_edge("flow_control", "error_handler")
        self.graph.add_edge("rag_retrieval", "error_handler")
        self.graph.add_edge("detailed_analysis", "error_handler")
        self.graph.add_edge("llm_analysis", "error_handler")
    
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
    
    async def execute(self, initial_state: ChargingAnalysisState) -> ChargingAnalysisState:
        """执行工作流"""
        final_state = await self.graph.ainvoke(initial_state)
        return final_state
```

### 5.2 进度回调机制

```python
class ProgressCallback:
    def __init__(self, websocket_manager, task_id: str):
        self.websocket_manager = websocket_manager
        self.task_id = task_id
        self.steps = [
            "文件验证中...",
            "报文解析中...", 
            "流程分析中...",
            "知识检索中...",
            "信号验证中...",
            "生成报告..."
        ]
    
    async def update_progress(self, step_index: int, message: str = None):
        progress = (step_index + 1) / len(self.steps) * 100
        
        await self.websocket_manager.send_to_task(
            self.task_id,
            {
                "type": "progress_update",
                "progress": progress,
                "current_step": message or self.steps[min(step_index, len(self.steps)-1)],
                "timestamp": datetime.now().isoformat()
            }
        )
```

## 6. RAG系统集成

### 6.1 向量数据库管理

```python
class VectorDatabaseManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
        self.embedding_model = self._load_embedding_model()
    
    def _load_embedding_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('BAAI/bge-base-zh-v1.5')
    
    async def create_collection(self, name: str, metadata: Dict[str, Any]):
        """创建知识库集合"""
        collection = self.client.create_collection(
            name=name,
            metadata=metadata
        )
        return collection
    
    async def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        """添加文档到集合"""
        collection = self.client.get_collection(collection_name)
        
        texts = [doc['content'] for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        # 分块处理长文档
        chunked_docs = []
        for doc in documents:
            chunks = self._chunk_document(doc['content'])
            for chunk in chunks:
                chunked_docs.append({
                    'content': chunk,
                    'metadata': doc['metadata'],
                    'source': doc.get('source', ''),
                    'chunk_id': len(chunked_docs)
                })
        
        # 批量添加
        collection.add(
            embeddings=[self.embedding_model.encode(doc['content']) for doc in chunked_docs],
            documents=[doc['content'] for doc in chunked_docs],
            metadatas=[doc['metadata'] for doc in chunked_docs],
            ids=[f"{collection_name}_{i}" for i in range(len(chunked_docs))]
        )
        
        return len(chunked_docs)
    
    async def search(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索文档"""
        collection = self.client.get_collection(collection_name)
        
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return [
            {
                'content': doc,
                'metadata': metadata,
                'score': 1 - distance  # 转换为相似度分数
            }
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )
        ]
    
    def _chunk_document(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """文档分块"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # 在句子边界处分割
            if end < len(content):
                sentence_end = content.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 设置下一个块的起始位置，包含重叠
            start = max(start + chunk_size - overlap, end)
        
        return chunks
```

### 6.2 知识库管理系统

```python
class KnowledgeBaseManager:
    def __init__(self):
        self.vector_db = VectorDatabaseManager()
        self.document_processor = DocumentProcessor()
    
    async def create_knowledge_base(self, name: str, description: str, 
                                   collection_type: str = "document") -> Dict[str, Any]:
        """创建知识库"""
        collection_metadata = {
            "name": name,
            "description": description,
            "type": collection_type,
            "created_at": datetime.now().isoformat(),
            "embedding_model": "bge-base-zh-v1.5"
        }
        
        collection = await self.vector_db.create_collection(name, collection_metadata)
        
        return {
            "collection_id": collection.name,
            "name": name,
            "description": description,
            "type": collection_type,
            "document_count": 0,
            "metadata": collection_metadata
        }
    
    async def upload_document(self, collection_name: str, file_path: str, 
                             metadata: Dict[str, Any]) -> Dict[str, Any]:
        """上传文档到知识库"""
        # 处理文档
        processed_doc = await self.document_processor.process_document(file_path)
        
        # 添加到向量数据库
        document_count = await self.vector_db.add_documents(
            collection_name, 
            [processed_doc]
        )
        
        return {
            "status": "completed",
            "document_count": document_count,
            "file_name": os.path.basename(file_path),
            "processing_time": processed_doc.get('processing_time', 0)
        }
    
    async def search_knowledge(self, collection_name: str, query: str, 
                              top_k: int = 5, min_score: float = 0.7) -> Dict[str, Any]:
        """搜索知识库"""
        results = await self.vector_db.search(collection_name, query, top_k)
        
        # 过滤低质量结果
        filtered_results = [r for r in results if r['score'] >= min_score]
        
        return {
            "query": query,
            "results": filtered_results,
            "total_found": len(filtered_results),
            "search_time": 0  # TODO: 记录搜索时间
        }
```

## 7. 训练系统架构

### 7.1 训练任务管理器

```python
class TrainingTaskManager:
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, TrainingTask] = {}
    
    async def create_training_task(self, task_config: Dict[str, Any]) -> str:
        """创建训练任务"""
        task_id = f"training_{int(time.time())}"
        
        task = TrainingTask(
            task_id=task_id,
            name=task_config['name'],
            dataset_id=task_config['dataset_id'],
            model_type=task_config['model_type'],
            hyperparameters=task_config['hyperparameters'],
            created_by=task_config['user_id']
        )
        
        # 保存任务到数据库
        await self._save_task_to_db(task)
        
        # 添加到队列
        await self.task_queue.put(task)
        
        return task_id
    
    async def start_training(self, task_id: str) -> Dict[str, Any]:
        """开始训练任务"""
        task = await self._load_task_from_db(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")
        
        # 更新任务状态
        task.status = "running"
        task.start_time = datetime.now()
        await self._update_task_status(task_id, task)
        
        # 异步执行训练
        asyncio.create_task(self._execute_training(task))
        
        return {
            "task_id": task_id,
            "status": "started",
            "estimated_duration": self._estimate_training_duration(task)
        }
    
    async def _execute_training(self, task: 'TrainingTask'):
        """执行训练任务"""
        try:
            self.active_tasks[task.task_id] = task
            
            # 加载数据集
            dataset = await self._load_dataset(task.dataset_id)
            
            # 初始化训练器
            trainer = await self.model_trainer.initialize(
                task.model_type,
                task.hyperparameters
            )
            
            # 设置进度回调
            progress_callback = TrainingProgressCallback(task.task_id)
            
            # 开始训练
            result = await trainer.train(
                dataset,
                callbacks=[progress_callback]
            )
            
            # 训练完成
            task.status = "completed"
            task.end_time = datetime.now()
            task.model_path = result['model_path']
            task.metrics = result['metrics']
            
            await self._update_task_status(task.task_id, task)
            
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            await self._update_task_status(task.task_id, task)
            
        finally:
            self.active_tasks.pop(task.task_id, None)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = await self._load_task_from_db(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status,
            "progress": task.progress,
            "current_epoch": task.current_epoch,
            "total_epochs": task.total_epochs,
            "metrics": task.metrics,
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
            "error_message": task.error_message
        }
```

### 7.2 模型训练器

```python
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.training_configs = {}
    
    async def initialize(self, model_type: str, hyperparameters: Dict[str, Any]):
        """初始化模型训练器"""
        if model_type == "flow_control":
            return FlowControlModelTrainer(hyperparameters)
        elif model_type == "llm":
            return LLMModelTrainer(hyperparameters)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

class FlowControlModelTrainer:
    def __init__(self, hyperparameters: Dict[str, Any]):
        self.hyperparameters = hyperparameters
        self.model = None
        self.tokenizer = None
    
    async def train(self, dataset, callbacks: List = None):
        """训练流程控制模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from torch.utils.data import DataLoader
        
        # 加载基础模型
        model_path = config.SMALL_MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 准备训练参数
        training_args = TrainingArguments(
            output_dir="./training_output",
            num_train_epochs=self.hyperparameters.get('epochs', 10),
            per_device_train_batch_size=self.hyperparameters.get('batch_size', 4),
            learning_rate=self.hyperparameters.get('learning_rate', 1e-4),
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # 准备数据集
        train_dataset = self._prepare_dataset(dataset)
        eval_dataset = self._split_eval_dataset(train_dataset)
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        model_path = "./models/flow_control_v1"
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        return {
            "model_path": model_path,
            "metrics": trainer.state.log_history[-1] if trainer.state.log_history else {}
        }
    
    def _prepare_dataset(self, dataset):
        """准备训练数据集"""
        # TODO: 实现数据集处理逻辑
        pass
    
    def _split_eval_dataset(self, dataset):
        """分割验证集"""
        # TODO: 实现数据集分割逻辑
        pass
```

### 7.3 训练进度监控

```python
class TrainingProgressCallback(TrainerCallback):
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.websocket_manager = WebSocketManager()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时的回调"""
        metrics = kwargs.get('logs', {})
        
        progress_data = {
            "task_id": self.task_id,
            "type": "training_progress",
            "current_epoch": state.epoch,
            "total_epochs": args.num_train_epochs,
            "step": state.global_step,
            "total_steps": state.max_steps,
            "progress": (state.epoch / args.num_train_epochs) * 100,
            "metrics": {
                "loss": metrics.get('loss'),
                "learning_rate": metrics.get('learning_rate'),
                "eval_loss": metrics.get('eval_loss')
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # 发送进度更新
        asyncio.create_task(
            self.websocket_manager.send_to_task(self.task_id, progress_data)
        )
        
        # 保存到数据库
        asyncio.create_task(self._save_metrics(progress_data))
    
    async def _save_metrics(self, progress_data):
        """保存训练指标到数据库"""
        # TODO: 实现指标保存逻辑
        pass
```

这个LangGraph架构设计提供了完整的充电分析工作流实现，包括状态管理、节点定义、流程控制和错误处理机制。