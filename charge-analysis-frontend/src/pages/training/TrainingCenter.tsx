import React from 'react';
import {
  Layout,
  Menu,
  Card,
  Form,
  Input,
  InputNumber,
  Select,
  Button,
  Space,
  Typography,
  Tag,
  Progress,
  Table,
  List,
  message,
  Divider,
  Empty,
  Switch,
  Descriptions,
  Statistic,
  Spin
} from 'antd';
import {
  SettingOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  AuditOutlined,
  CloudUploadOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import ReactECharts from 'echarts-for-react';

import { useAuthStore } from '../../stores/authStore';
import {
  trainingService,
  TrainingConfig,
  TrainingTaskDetail,
  TrainingMetricPoint,
  TrainingLogEntry,
  TrainingEvaluation,
  KeywordEvalResult
} from '../../services/trainingService';

const { Content, Sider } = Layout;
const { Title, Text } = Typography;

const menuItems = [
  { key: 'config', icon: <SettingOutlined />, label: '训练配置' },
  { key: 'execution', icon: <ThunderboltOutlined />, label: '训练执行' },
  { key: 'progress', icon: <LineChartOutlined />, label: '训练进度' },
  { key: 'evaluation', icon: <AuditOutlined />, label: '训练评估' },
  { key: 'publish', icon: <CloudUploadOutlined />, label: '模型发布' }
];

const TrainingCenter: React.FC = () => {
  const { token } = useAuthStore();
  const [activeMenu, setActiveMenu] = React.useState('config');
  const [configs, setConfigs] = React.useState<TrainingConfig[]>([]);
  const [tasks, setTasks] = React.useState<TrainingTaskDetail[]>([]);
  const [selectedTaskId, setSelectedTaskId] = React.useState<number | null>(null);
  const [selectedTask, setSelectedTask] = React.useState<TrainingTaskDetail | null>(null);
  const [metrics, setMetrics] = React.useState<TrainingMetricPoint[]>([]);
  const [logs, setLogs] = React.useState<TrainingLogEntry[]>([]);
  const [evaluation, setEvaluation] = React.useState<TrainingEvaluation | null>(null);
  const [datasetName, setDatasetName] = React.useState('');
  const [datasetFile, setDatasetFile] = React.useState<File | null>(null);
  const [datasetInfo, setDatasetInfo] = React.useState<{ id: number | null; sampleCount: number | null }>({
    id: null,
    sampleCount: null
  });
  const [uploadingDataset, setUploadingDataset] = React.useState(false);
  const [savingConfig, setSavingConfig] = React.useState(false);
  const [creatingTask, setCreatingTask] = React.useState(false);
  const [logsLoading, setLogsLoading] = React.useState(false);
  const [metricsLoading, setMetricsLoading] = React.useState(false);
  const [publishing, setPublishing] = React.useState(false);
  const [loadingTaskDetail, setLoadingTaskDetail] = React.useState(false);

  const [yamlText, setYamlText] = React.useState<string>(`### model
model_name_or_path: /opt/app/models/deepseek-7B

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### dataset
dataset: "2015"
template: qwen
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/deepseek-r1-qwen1.5b/lora/sft
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-4
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 50
`);
  const [yamlFile, setYamlFile] = React.useState<File | null>(null);
  const [parsedYaml, setParsedYaml] = React.useState<Record<string, any> | null>(null);
  const [parsingYaml, setParsingYaml] = React.useState(false);

  const [evalFile, setEvalFile] = React.useState<File | null>(null);
  const [keywordEvalResult, setKeywordEvalResult] = React.useState<KeywordEvalResult | null>(null);
  const [runningKeywordEval, setRunningKeywordEval] = React.useState(false);

  const [configForm] = Form.useForm();
  const [taskForm] = Form.useForm();
  const [evaluationForm] = Form.useForm();
  const [publishForm] = Form.useForm();

  const handleDatasetFileChange = (file: File | null) => {
    if (file) {
      setDatasetFile(file);
      if (!datasetName) {
        const guessedName = file.name.replace(/\.[^/.]+$/, '');
        setDatasetName(guessedName);
      }
    } else {
      setDatasetFile(null);
    }
  };

  const handleYamlFileChange = async (file: File | null) => {
    setYamlFile(file);
    if (!file) return;
    try {
      const text = await file.text();
      setYamlText(text);
      message.success('已载入 YAML 文件内容');
    } catch (e: any) {
      message.error(e?.message || '读取 YAML 文件失败');
    }
  };

  const handleParseYaml = async () => {
    if (!ensureToken()) return;
    if (!yamlText.trim()) {
      message.warning('请先输入 YAML 内容');
      return;
    }
    setParsingYaml(true);
    try {
      const cfg = await trainingService.parseYamlConfig(yamlText, token!);
      setParsedYaml(cfg);
      message.success('YAML 解析成功');
    } catch (e: any) {
      message.error(e?.message || 'YAML 解析失败');
    } finally {
      setParsingYaml(false);
    }
  };

  const handleLoadConfig = (config: TrainingConfig) => {
    configForm.setFieldsValue({
      configId: config.id,
      name: config.name,
      baseModel: config.baseModel,
      modelPath: config.modelPath,
      modelSize: config.modelSize,
      datasetStrategy: config.datasetStrategy,
      hyperparameters: config.hyperparameters,
      notes: config.notes
    });
    message.success(`已载入配置「${config.name}」`);
  };

  const handleTaskConfigChange = (configId?: number | null) => {
    if (!configId) {
      return;
    }
    const config = configs.find((item) => item.id === configId);
    if (!config) {
      return;
    }
    const hyper = (config.hyperparameters || {}) as Record<string, unknown>;
    const pickNumber = (...keys: string[]) => {
      for (const key of keys) {
        const value = hyper[key];
        if (typeof value === 'number') {
          return value;
        }
        if (typeof value === 'string' && value.trim() !== '' && !Number.isNaN(Number(value))) {
          return Number(value);
        }
      }
      return undefined;
    };
    taskForm.setFieldsValue({
      modelSize: config.modelSize,
      epochs: pickNumber('epochs', 'Epochs') ?? taskForm.getFieldValue('epochs'),
      batchSize: pickNumber('batch_size', 'batchSize', 'BatchSize') ?? taskForm.getFieldValue('batchSize'),
      learningRate: pickNumber('learning_rate', 'learningRate', 'LearningRate') ?? taskForm.getFieldValue('learningRate')
    });
    message.success(`已根据「${config.name}」同步超参数`);
  };

  const ensureToken = () => {
    if (!token) {
      message.warning('请先登录以使用训练功能');
      return false;
    }
    return true;
  };

  const loadConfigs = React.useCallback(async () => {
    if (!token) return;
    try {
      const list = await trainingService.fetchConfigs(token);
      setConfigs(list);
    } catch (error: any) {
      message.error(error.message || '加载训练配置失败');
    }
  }, [token]);

  const loadTasks = React.useCallback(async () => {
    if (!token) return;
    try {
      const list = await trainingService.listTasks(token);
      setTasks(list);
      if (list.length && !selectedTaskId) {
        setSelectedTaskId(list[0].id);
      }
    } catch (error: any) {
      message.error(error.message || '加载训练任务失败');
    }
  }, [token, selectedTaskId]);

  const loadTaskStreams = React.useCallback(
    async (taskId: number) => {
      if (!token) return;
      try {
        setMetricsLoading(true);
        setLogsLoading(true);
        const [metricsData, logsData, evaluationData] = await Promise.all([
          trainingService.getTaskMetrics(taskId, token),
          trainingService.getTaskLogs(taskId, token),
          trainingService.getTaskEvaluation(taskId, token)
        ]);
        setMetrics(metricsData);
        setLogs(logsData);
        setEvaluation(evaluationData);
      } catch (error: any) {
        message.error(error.message || '加载训练进度失败');
      } finally {
        setMetricsLoading(false);
        setLogsLoading(false);
      }
    },
    [token]
  );

  const loadTaskDetail = React.useCallback(
    async (taskId?: number, refreshStreams = true) => {
      if (!token) return;
      const currentId = taskId ?? selectedTaskId;
      if (!currentId) return;
      try {
        setLoadingTaskDetail(true);
        const detail = await trainingService.getTask(currentId, token);
        setSelectedTask(detail);
        if (refreshStreams) {
          loadTaskStreams(currentId);
        }
      } catch (error: any) {
        message.error(error.message || '获取任务详情失败');
      } finally {
        setLoadingTaskDetail(false);
      }
    },
    [token, selectedTaskId, loadTaskStreams]
  );

  React.useEffect(() => {
    if (!token) return;
    loadConfigs();
    loadTasks();
  }, [token, loadConfigs, loadTasks]);

  React.useEffect(() => {
    if (!selectedTaskId || !token) return;
    loadTaskDetail(selectedTaskId);
  }, [selectedTaskId, token, loadTaskDetail]);

  React.useEffect(() => {
    if (!token || !selectedTaskId) return;
    const interval = window.setInterval(() => {
      loadTaskDetail(selectedTaskId, false);
      loadTaskStreams(selectedTaskId);
    }, 5000);
    return () => window.clearInterval(interval);
  }, [token, selectedTaskId, loadTaskDetail, loadTaskStreams]);

  const handleDatasetUpload = async () => {
    if (!ensureToken() || !datasetFile || !datasetName) return;
    setUploadingDataset(true);
    try {
      message.loading({ content: '正在上传数据集...', key: 'datasetUpload' });
      const result = await trainingService.uploadDataset(datasetName, datasetFile, token!, {
        description: '训练数据集',
        datasetType: 'standard'
      });
      setDatasetInfo({ id: result.id, sampleCount: result.sampleCount });
      message.success({ content: '数据集上传成功', key: 'datasetUpload' });
    } catch (error: any) {
      message.error(error.message || '上传失败');
    } finally {
      setUploadingDataset(false);
    }
  };

  const handleSaveConfig = async (values: any) => {
    if (!ensureToken()) return;
    setSavingConfig(true);
    try {
      const payload = {
        name: values.name,
        baseModel: values.baseModel,
        modelPath: values.modelPath,
        adapterType: 'lora' as const,
        modelSize: values.modelSize,
        datasetStrategy: values.datasetStrategy,
        hyperparameters: values.hyperparameters,
        notes: values.notes
      };
      if (values.configId) {
        await trainingService.updateConfig(values.configId, payload, token!);
        message.success('配置已更新');
      } else {
        await trainingService.saveConfig(payload, token!);
        message.success('配置已保存');
      }
      configForm.resetFields(['configId']);
      loadConfigs();
    } catch (error: any) {
      message.error(error.message || '保存配置失败');
    } finally {
      setSavingConfig(false);
    }
  };

  const handleCreateTask = async (values: any) => {
    if (!ensureToken()) return;
    if (!datasetInfo.id) {
      message.warning('请先上传训练数据集');
      return;
    }
    setCreatingTask(true);
    try {
      const useYaml = yamlText.trim().length > 0;
      const task = useYaml
        ? await trainingService.createSftLoraTask(
            {
              name: values.taskName,
              description: values.description,
              datasetId: datasetInfo.id,
              modelType: 'flow_control',
              yamlText
            },
            token!
          )
        : await trainingService.createTrainingTask(
            {
              name: values.taskName,
              description: values.description,
              datasetId: datasetInfo.id,
              configId: values.configId,
              modelType: 'flow_control',
              modelSize: values.modelSize,
              hyperparameters: {
                epochs: values.epochs,
                batch_size: values.batchSize,
                learning_rate: values.learningRate
              }
            },
            token!
          );
      await trainingService.startTraining(task.id, token!);
      message.success('训练任务已创建并启动');
      taskForm.resetFields(['taskName', 'description']);
      loadTasks();
    } catch (error: any) {
      message.error(error.message || '创建训练任务失败');
    } finally {
      setCreatingTask(false);
    }
  };

  const handleStartTask = async (taskId: number) => {
    if (!ensureToken()) return;
    try {
      await trainingService.startTraining(taskId, token!);
      message.success('任务已加入队列');
      loadTaskDetail(taskId);
      loadTasks();
    } catch (error: any) {
      message.error(error.message || '启动任务失败');
    }
  };

  const handleEvaluationSubmit = async (values: any) => {
    if (!ensureToken() || !selectedTaskId) return;
    try {
      const metricsPayload = {
        accuracy: selectedTask?.metrics?.accuracy,
        loss: selectedTask?.metrics?.loss,
        throughput: values.throughput,
        latency: values.latency
      };
      const result = await trainingService.evaluateTask(
        selectedTaskId,
        {
          evaluationType: values.evaluationType,
          metrics: metricsPayload,
          recommendedPlan: values.recommendedPlan,
          notes: values.notes
        },
        token!
      );
      setEvaluation(result);
      message.success('评估记录已更新');
    } catch (error: any) {
      message.error(error.message || '评估提交失败');
    }
  };

  const handlePublish = async (values: any) => {
    if (!ensureToken()) return;
    const targetTaskId = values.taskId || selectedTaskId;
    if (!targetTaskId) {
      message.warning('请选择要发布的训练任务');
      return;
    }
    setPublishing(true);
    try {
      await trainingService.publishModel(
        targetTaskId,
        {
          version: values.version,
          targetEnvironment: values.targetEnvironment,
          endpointUrl: values.endpointUrl,
          notes: values.notes,
          setDefault: values.setDefault
        },
        token!
      );
      message.success('模型已发布');
      setSelectedTaskId(targetTaskId);
      publishForm.resetFields();
    } catch (error: any) {
      message.error(error.message || '模型发布失败');
    } finally {
      setPublishing(false);
    }
  };

  const handleRunKeywordEval = async () => {
    if (!ensureToken() || !selectedTaskId) return;
    if (!evalFile) {
      message.warning('请先选择评估集 JSON 文件');
      return;
    }
    setRunningKeywordEval(true);
    try {
      const result = await trainingService.evaluateKeywords(selectedTaskId, evalFile, token!);
      setKeywordEvalResult(result);
      message.success('关键词评估完成');
      // 同步刷新“评估记录”（后端会写入 TrainingEvaluation）
      loadTaskStreams(selectedTaskId);
    } catch (e: any) {
      message.error(e?.message || '关键词评估失败');
    } finally {
      setRunningKeywordEval(false);
    }
  };

  const renderConfigPanel = () => (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <div>
        <Title level={3}>配置训练模板</Title>
        <Text type="secondary">预设底座模型、LoRA 适配器与关键超参数，便于快速复用。</Text>
      </div>
      <Card>
        <Form
          layout="vertical"
          form={configForm}
          initialValues={{
            modelSize: '1.5b',
            datasetStrategy: 'full',
            hyperparameters: { epochs: 10, batch_size: 32, learning_rate: 0.001 }
          }}
          onFinish={handleSaveConfig}
        >
          <Form.Item name="configId" hidden>
            <Input />
          </Form.Item>
          <Form.Item label="配置名称" name="name" rules={[{ required: true, message: '请输入配置名称' }]}> 
            <Input placeholder="如：1.5B-LoRA-基线" />
          </Form.Item>
          <Form.Item label="底座模型" name="baseModel" rules={[{ required: true }]}> 
            <Input placeholder="如：internlm2-1_5b" />
          </Form.Item>
          <Form.Item label="模型存储路径" name="modelPath" rules={[{ required: true }]}> 
            <Input placeholder="/models/base/1.5b" />
          </Form.Item>
          <Form.Item label="模型规模" name="modelSize" rules={[{ required: true }]}> 
            <Select options={[{ value: '1.5b', label: '1.5B' }, { value: '7b', label: '7B' }]} />
          </Form.Item>
          <Form.Item label="数据策略" name="datasetStrategy"> 
            <Select
              options={[
                { value: 'full', label: '全量训练' },
                { value: 'balanced', label: '均衡采样' },
                { value: 'high_quality', label: '高质量优先' }
              ]}
            />
          </Form.Item>
          <Card size="small" title="默认超参数" style={{ marginBottom: 16 }}>
            <Form.Item label="Epochs" name={[ 'hyperparameters', 'epochs' ]} rules={[{ required: true }]}>
              <InputNumber min={1} max={30} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="Batch Size" name={[ 'hyperparameters', 'batch_size' ]} rules={[{ required: true }]}> 
              <InputNumber min={4} max={128} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="Learning Rate" name={[ 'hyperparameters', 'learning_rate' ]} rules={[{ required: true }]}> 
              <InputNumber min={0.0001} max={0.01} step={0.0001} style={{ width: '100%' }} />
            </Form.Item>
          </Card>
          <Form.Item label="备注" name="notes">
            <Input.TextArea rows={3} placeholder="记录使用场景或硬件要求" />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={savingConfig}>
            保存配置
          </Button>
        </Form>
      </Card>
      <Card title="已保存配置" bodyStyle={{ maxHeight: 320, overflow: 'auto' }}>
        {configs.length === 0 ? (
          <Empty description="暂无配置" />
        ) : (
          <List
            dataSource={configs}
            renderItem={(item) => (
              <List.Item
                key={item.id}
                actions={[
                  <Button type="link" onClick={() => handleLoadConfig(item)}>
                    载入
                  </Button>
                ]}
              >
                <List.Item.Meta
                  title={
                    <Space>
                      <span>{item.name}</span>
                      <Tag color="blue">{item.modelSize.toUpperCase()}</Tag>
                    </Space>
                  }
                  description={`底座模型：${item.baseModel}`}
                />
                <div>{dayjs(item.updatedAt).format('MM-DD HH:mm')}</div>
              </List.Item>
            )}
          />
        )}
      </Card>
    </Space>
  );

  const renderExecutionPanel = () => (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card title="上传训练数据集">
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Input
            placeholder="数据集名称"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
          />
          <input
            type="file"
            accept=".json,.jsonl,.csv,.xlsx"
            onChange={(e) => handleDatasetFileChange(e.target.files?.[0] || null)}
          />
          <Button
            type="primary"
            onClick={handleDatasetUpload}
            disabled={!datasetFile || !datasetName}
            loading={uploadingDataset}
          >
            上传数据集
          </Button>
          {datasetInfo.id && (
            <Tag color="success">数据集 ID：{datasetInfo.id}（样本 {datasetInfo.sampleCount}）</Tag>
          )}
        </Space>
      </Card>
      <Card title="YAML 配置（SFT + LoRA，最小化 llamafactory-cli train）">
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Text type="secondary">
            支持手动粘贴或上传 YAML。创建任务时若 YAML 非空，将优先使用 YAML 一键训练。
          </Text>
          <input
            type="file"
            accept=".yaml,.yml"
            onChange={(e) => handleYamlFileChange(e.target.files?.[0] || null)}
          />
          <Input.TextArea
            rows={14}
            value={yamlText}
            onChange={(e) => setYamlText(e.target.value)}
            placeholder="粘贴/输入 llamafactory 风格的 YAML 配置"
          />
          <Space>
            <Button onClick={handleParseYaml} loading={parsingYaml}>
              解析 YAML
            </Button>
            {parsedYaml && <Tag color="blue">已解析：output_dir={(parsedYaml.output_dir as any) ?? '-'}</Tag>}
          </Space>
        </Space>
      </Card>
      <Card title="创建训练任务">
        <Form
          layout="vertical"
          form={taskForm}
          initialValues={{
            modelSize: '1.5b',
            epochs: 10,
            batchSize: 32,
            learningRate: 0.001
          }}
          onFinish={handleCreateTask}
        >
          <Form.Item
            label="任务名称"
            name="taskName"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="如：LoRA-1.5B-场景A" />
          </Form.Item>
          <Form.Item label="描述" name="description">
            <Input.TextArea rows={2} placeholder="可选" />
          </Form.Item>
          <Form.Item label="关联配置" name="configId">
            <Select
              allowClear
              placeholder="选择训练配置"
              options={configs.map((cfg) => ({ value: cfg.id, label: `${cfg.name} (${cfg.modelSize.toUpperCase()})` }))}
              onChange={(value) => handleTaskConfigChange(typeof value === 'number' ? value : null)}
            />
          </Form.Item>
          <Form.Item label="模型规模" name="modelSize" rules={[{ required: true }]}> 
            <Select options={[{ value: '1.5b', label: '1.5B' }, { value: '7b', label: '7B' }]} />
          </Form.Item>
          <Form.Item label="Epochs" name="epochs" rules={[{ required: true }]}>
            <InputNumber min={1} max={30} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item label="Batch Size" name="batchSize" rules={[{ required: true }]}>
            <InputNumber min={4} max={128} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item label="Learning Rate" name="learningRate" rules={[{ required: true }]}>
            <InputNumber min={0.0001} max={0.01} step={0.0001} style={{ width: '100%' }} />
          </Form.Item>
          <Tag color={yamlText.trim() ? 'green' : 'default'}>
            {yamlText.trim() ? '将使用 YAML 一键训练（SFT+LoRA）' : '未提供 YAML：将使用旧版简化参数（模拟/兼容）'}
          </Tag>
          <Button type="primary" htmlType="submit" loading={creatingTask}>
            创建并启动训练
          </Button>
        </Form>
      </Card>
      <Card title="训练任务列表">
        <Table
          dataSource={tasks}
          rowKey="id"
          pagination={false}
          columns={[
            { title: '任务', dataIndex: 'name' },
            {
              title: '规模',
              dataIndex: 'modelSize',
              render: (val: string) => <Tag>{val.toUpperCase()}</Tag>
            },
            {
              title: '状态',
              dataIndex: 'status',
              render: (status: string) => (
                <Tag color={status === 'completed' ? 'green' : status === 'running' ? 'blue' : 'default'}>
                  {status}
                </Tag>
              )
            },
            {
              title: '进度',
              dataIndex: 'progress',
              render: (value: number) => <Progress percent={Math.round(value)} size="small" status="active" />
            },
            {
              title: '操作',
              render: (_, record) => (
                <Space>
                  <Button type="link" onClick={() => setSelectedTaskId(record.id)}>
                    查看
                  </Button>
                  {record.status === 'pending' && (
                    <Button type="link" onClick={() => handleStartTask(record.id)}>
                      启动
                    </Button>
                  )}
                </Space>
              )
            }
          ]}
        />
      </Card>
    </Space>
  );

  const chartOptions = React.useMemo(() => {
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: ['Loss', 'Accuracy'] },
      xAxis: {
        type: 'category',
        data: metrics.map((item) => `E${item.epoch}-S${item.step}`)
      },
      yAxis: [
          { type: 'value', name: 'Loss', min: 0 },
          { type: 'value', name: 'Accuracy', min: 0, max: 1, position: 'right' }
      ],
      series: [
        {
          name: 'Loss',
          type: 'line',
          data: metrics.map((item) => item.loss ?? null),
          smooth: true
        },
        {
          name: 'Accuracy',
          type: 'line',
          yAxisIndex: 1,
          data: metrics.map((item) => item.accuracy ?? null),
          smooth: true
        }
      ]
    };
  }, [metrics]);

  const renderProgressPanel = () => {
    if (!selectedTask) {
      return <Empty description="请选择训练任务" />;
    }
    return (
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Card title={`${selectedTask.name} - 进度`} loading={loadingTaskDetail}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Space align="center" size="large">
              <Progress type="dashboard" percent={Math.round(selectedTask.progress)} />
              <Descriptions column={2} size="small" bordered>
                <Descriptions.Item label="模型规模">{selectedTask.modelSize.toUpperCase()}</Descriptions.Item>
                <Descriptions.Item label="状态">{selectedTask.status}</Descriptions.Item>
                <Descriptions.Item label="当前 Epoch">{selectedTask.currentEpoch || '-'}</Descriptions.Item>
                <Descriptions.Item label="总 Epoch">{selectedTask.totalEpochs || '-'}</Descriptions.Item>
              </Descriptions>
            </Space>
            <Divider />
            <div>
              <Title level={5}>关键曲线</Title>
              {metricsLoading ? (
                <Spin />
              ) : metrics.length === 0 ? (
                <Empty description="暂无指标" />
              ) : (
                <ReactECharts option={chartOptions} style={{ height: 320 }} />
              )}
            </div>
            <Divider />
            <div>
              <Title level={5}>最近日志</Title>
              <List
                loading={logsLoading}
                dataSource={logs.slice(0, 8)}
                renderItem={(item) => (
                  <List.Item key={item.id}>
                    <Space direction="vertical" size={0} style={{ width: '100%' }}>
                      <Space>
                        <Tag>{item.logLevel}</Tag>
                        <Text type="secondary">{dayjs(item.createdAt).format('HH:mm:ss')}</Text>
                      </Space>
                      <Text>{item.message}</Text>
                    </Space>
                  </List.Item>
                )}
              />
            </div>
          </Space>
        </Card>
      </Space>
    );
  };

  const renderEvaluationPanel = () => {
    if (!selectedTask) {
      return <Empty description="请选择训练任务" />;
    }
    return (
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Card title="关键词评估（上传 JSON 评估集）">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Text type="secondary">
              评估集格式：<Text code>[{`{"question":"...","expected_keywords":["..."]}`}]</Text>
            </Text>
            <input type="file" accept=".json" onChange={(e) => setEvalFile(e.target.files?.[0] || null)} />
            <Button type="primary" onClick={handleRunKeywordEval} loading={runningKeywordEval}>
              运行关键词评估
            </Button>
            {keywordEvalResult && (
              <Card size="small" title="评估摘要">
                <Space size="large">
                  <Statistic title="总题数" value={keywordEvalResult.total} />
                  <Statistic title="严格通过率" value={keywordEvalResult.strict_pass_rate} />
                  <Statistic title="平均命中率" value={keywordEvalResult.avg_hit_rate} />
                </Space>
                <Divider />
                <Table
                  rowKey={(r) => r.question}
                  dataSource={keywordEvalResult.details}
                  pagination={{ pageSize: 5 }}
                  columns={[
                    { title: '问题', dataIndex: 'question' },
                    {
                      title: '命中率',
                      dataIndex: 'hit_rate',
                      render: (v: number) => <Tag>{v}</Tag>
                    },
                    {
                      title: '严格通过',
                      dataIndex: 'strict_pass',
                      render: (v: boolean) => <Tag color={v ? 'green' : 'red'}>{String(v)}</Tag>
                    }
                  ]}
                  expandable={{
                    expandedRowRender: (r) => (
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text type="secondary">期望关键词：{r.expected_keywords?.join('、') || '-'}</Text>
                        <Text type="secondary">命中关键词：{r.hit_keywords?.join('、') || '-'}</Text>
                        <Divider style={{ margin: '8px 0' }} />
                        <Text>{r.answer}</Text>
                      </Space>
                    )
                  }}
                />
              </Card>
            )}
          </Space>
        </Card>
        <Card title="评估训练效果">
          <Form
            layout="vertical"
            form={evaluationForm}
            initialValues={{ evaluationType: 'automatic', recommendedPlan: 'balanced' }}
            onFinish={handleEvaluationSubmit}
          >
            <Form.Item label="评估方式" name="evaluationType">
              <Select
                options={[
                  { value: 'automatic', label: '自动评估' },
                  { value: 'manual', label: '人工评估' }
                ]}
              />
            </Form.Item>
            <Form.Item label="推荐方案" name="recommendedPlan" rules={[{ required: true }]}> 
              <Select
                options={[
                  { value: 'balanced', label: '均衡方案（精度+速度）' },
                  { value: 'accuracy_first', label: '精度优先' },
                  { value: 'latency_first', label: '时延优先' }
                ]}
              />
            </Form.Item>
            <Form.Item label="吞吐（samples/s）" name="throughput">
              <InputNumber min={1} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="时延（ms）" name="latency">
              <InputNumber min={1} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="备注" name="notes">
              <Input.TextArea rows={3} placeholder="记录结论或风险" />
            </Form.Item>
            <Button type="primary" htmlType="submit">
              保存评估
            </Button>
          </Form>
        </Card>
        <Card title="评估结果">
          {evaluation ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <Tag color="blue">{evaluation.evaluationType}</Tag>
              <Text>推荐方案：{evaluation.recommendedPlan}</Text>
              <Text>评估人：{evaluation.evaluator || '系统'}</Text>
              <Text type="secondary">{dayjs(evaluation.createdAt).format('YYYY-MM-DD HH:mm')}</Text>
              <Card size="small" title="指标摘要">
                {Object.entries(evaluation.metrics || {}).map(([key, value]) => (
                  <div key={key} style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text type="secondary">{key}</Text>
                    <Text>{value as any}</Text>
                  </div>
                ))}
              </Card>
              {evaluation.notes && <Text>{evaluation.notes}</Text>}
            </Space>
          ) : (
            <Empty description="尚未评估" />
          )}
        </Card>
      </Space>
    );
  };

  const renderPublishPanel = () => {
    const completedTasks = tasks.filter((task) => task.status === 'completed');
    if (completedTasks.length === 0) {
      return <Empty description="暂无可发布模型" />;
    }
    return (
      <Card title="发布训练模型">
        <Form layout="vertical" form={publishForm} onFinish={handlePublish}>
          <Form.Item label="选择任务" name="taskId" rules={[{ required: true, message: '请选择任务' }]}> 
            <Select
              placeholder="选择已完成任务"
              onChange={(value: number) => setSelectedTaskId(value)}
              options={completedTasks.map((task) => ({ value: task.id, label: `${task.name} (${task.modelSize.toUpperCase()})` }))}
            />
          </Form.Item>
          <Form.Item label="版本号" name="version" rules={[{ required: true, message: '请输入版本号' }]}> 
            <Input placeholder="v1.0.0" />
          </Form.Item>
          <Form.Item label="目标环境" name="targetEnvironment" rules={[{ required: true }]}> 
            <Input placeholder="如：edge-a100" />
          </Form.Item>
          <Form.Item label="服务地址" name="endpointUrl">
            <Input placeholder="可选，例如 http://host/api" />
          </Form.Item>
          <Form.Item label="备注" name="notes">
            <Input.TextArea rows={3} />
          </Form.Item>
          <Form.Item label="设为默认" name="setDefault" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={publishing} icon={<CheckCircleOutlined />}>
            发布模型
          </Button>
        </Form>
      </Card>
    );
  };

  const renderContent = () => {
    switch (activeMenu) {
      case 'config':
        return renderConfigPanel();
      case 'execution':
        return renderExecutionPanel();
      case 'progress':
        return renderProgressPanel();
      case 'evaluation':
        return renderEvaluationPanel();
      case 'publish':
        return renderPublishPanel();
      default:
        return null;
    }
  };

  return (
    <Card style={{ minHeight: 'calc(100vh - 120px)', borderRadius: 12, border: '1px solid #e5e7eb' }}>
      <Layout style={{ background: 'transparent' }}>
        <Sider
          width={220}
          style={{
            background: '#f8fafc',
            borderRight: '1px solid #e2e8f0',
            borderRadius: 8,
            marginRight: 24
          }}
        >
          <div style={{ padding: '16px 24px' }}>
            <Title level={4} style={{ marginBottom: 8 }}>
              训练控制台
            </Title>
            <Text type="secondary">LoRA 1.5B / 7B 训练全流程</Text>
          </div>
          <Menu
            mode="inline"
            selectedKeys={[activeMenu]}
            items={menuItems}
            onClick={(info) => setActiveMenu(info.key)}
            style={{ borderRight: 'none' }}
          />
        </Sider>
        <Content>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            {tasks.length > 0 && (
              <Card>
                <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Statistic title="正在跟踪的任务" value={selectedTask?.name || '未选择'} />
                  <Select
                    style={{ minWidth: 260 }}
                    placeholder="切换任务"
                    value={selectedTaskId || undefined}
                    options={tasks.map((task) => ({ value: task.id, label: `${task.name} (${task.status})` }))}
                    onChange={(value) => setSelectedTaskId(value)}
                  />
                </Space>
              </Card>
            )}
            {renderContent()}
          </Space>
        </Content>
      </Layout>
    </Card>
  );
};

export default TrainingCenter;
