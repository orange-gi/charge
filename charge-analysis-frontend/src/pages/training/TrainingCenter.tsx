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
  TrainingEvaluation
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

  const [configForm] = Form.useForm();
  const [taskForm] = Form.useForm();
  const [evaluationForm] = Form.useForm();
  const [publishForm] = Form.useForm();

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
      const hyperparameters = {
        epochs: values.epochs,
        batch_size: values.batchSize,
        learning_rate: values.learningRate
      };
      const task = await trainingService.createTrainingTask(
        {
          name: values.taskName,
          description: values.description,
          datasetId: datasetInfo.id,
          configId: values.configId,
          modelType: 'flow_control',
          modelSize: values.modelSize,
          hyperparameters
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
                  <Button
                    type="link"
                    onClick={() =>
                      configForm.setFieldsValue({
                        configId: item.id,
                        name: item.name,
                        baseModel: item.baseModel,
                        modelPath: item.modelPath,
                        modelSize: item.modelSize,
                        datasetStrategy: item.datasetStrategy,
                        hyperparameters: item.hyperparameters,
                        notes: item.notes
                      })
                    }
                  >
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
            accept=".csv,.xlsx"
            onChange={(e) => setDatasetFile(e.target.files?.[0] || null)}
          />
          <Button type="primary" onClick={handleDatasetUpload} disabled={!datasetFile || !datasetName} loading={uploadingDataset}>
            上传数据集
          </Button>
          {datasetInfo.id && (
            <Tag color="success">数据集 ID：{datasetInfo.id}（样本 {datasetInfo.sampleCount}）</Tag>
          )}
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
