import React from 'react';
import {
  Layout,
  Menu,
  Card,
  Space,
  Typography,
  Button,
  Select,
  Input,
  Table,
  Tag,
  Modal,
  Form,
  message,
  Empty,
  Descriptions,
  Statistic,
  Row,
  Col,
  Spin
} from 'antd';
import { BarChartOutlined, DatabaseOutlined, SearchOutlined, UploadOutlined, PlusOutlined } from '@ant-design/icons';
import * as XLSX from 'xlsx';
import dayjs from 'dayjs';

import { useAuthStore } from '../../stores/authStore';
import {
  ragService,
  type KnowledgeCollection,
  type KnowledgeDocument,
  type RAGQueryRecord,
  type RagDocumentLog
} from '../../services/ragService';

const { Content, Sider } = Layout;
const { Title, Text } = Typography;

type MenuKey = 'board' | 'manage' | 'query';

type ExcelPreview = {
  filename: string;
  fileSize: number;
  sheetName: string;
  headers: string[];
  primaryTagKey: string;
  rowsTotal: number;
  rowsIndexed: number;
  rowsSkippedEmptyTag: number;
  primaryTagUnique: number;
};

function isExcelFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  return lower.endsWith('.xlsx') || lower.endsWith('.xlsm');
}

function formatBytes(bytes?: number | null): string {
  if (!bytes || bytes <= 0) return '-';
  const kb = 1024;
  const mb = kb * 1024;
  if (bytes >= mb) return `${(bytes / mb).toFixed(2)} MB`;
  if (bytes >= kb) return `${(bytes / kb).toFixed(1)} KB`;
  return `${bytes} B`;
}

async function parseExcelPreview(file: File): Promise<ExcelPreview> {
  const buf = await file.arrayBuffer();
  const wb = XLSX.read(buf, { type: 'array' });
  const sheetName = wb.SheetNames?.[0];
  if (!sheetName) {
    throw new Error('Excel 不包含任何 Sheet');
  }
  const ws = wb.Sheets[sheetName];
  const rows = XLSX.utils.sheet_to_json(ws, { header: 1, raw: false, blankrows: false }) as unknown as any[][];
  if (!rows || rows.length === 0) {
    throw new Error('Excel 第一行为空，无法识别表头');
  }

  const headerRow = rows[0] || [];
  let headers = headerRow.map((h) => String(h ?? '').trim());
  if (!headers.length || !headers[0]) {
    headers = headerRow.map((_, idx) => `列${idx + 1}`);
  }

  const primaryTagKey = headers[0] || '主标签';
  const dataRows = rows.slice(1);
  const rowsTotal = dataRows.length;

  let rowsIndexed = 0;
  let rowsSkippedEmptyTag = 0;
  const unique = new Set<string>();

  for (const row of dataRows) {
    const primary = row?.[0];
    const v = String(primary ?? '').trim();
    if (!v) {
      rowsSkippedEmptyTag += 1;
      continue;
    }
    rowsIndexed += 1;
    unique.add(v);
  }

  return {
    filename: file.name,
    fileSize: file.size,
    sheetName,
    headers,
    primaryTagKey,
    rowsTotal,
    rowsIndexed,
    rowsSkippedEmptyTag,
    primaryTagUnique: unique.size
  };
}

const menuItems = [
  { key: 'board', icon: <BarChartOutlined />, label: '知识库看板' },
  { key: 'manage', icon: <DatabaseOutlined />, label: '知识库管理' },
  { key: 'query', icon: <SearchOutlined />, label: '知识库查询' }
];

const RagCenter: React.FC = () => {
  const { token } = useAuthStore();

  const [activeMenu, setActiveMenu] = React.useState<MenuKey>('board');

  const [collections, setCollections] = React.useState<KnowledgeCollection[]>([]);
  const [collectionsLoading, setCollectionsLoading] = React.useState(false);
  const [collectionId, setCollectionId] = React.useState<number | null>(null);

  const [documents, setDocuments] = React.useState<KnowledgeDocument[]>([]);
  const [documentsLoading, setDocumentsLoading] = React.useState(false);

  const [queryHistory, setQueryHistory] = React.useState<RAGQueryRecord[]>([]);
  const [queryHistoryLoading, setQueryHistoryLoading] = React.useState(false);

  const [queryText, setQueryText] = React.useState('');
  const [isQuerying, setIsQuerying] = React.useState(false);
  const [answer, setAnswer] = React.useState('');
  const [resultDocs, setResultDocs] = React.useState<any[]>([]);
  const [queryRuntime, setQueryRuntime] = React.useState<number | null>(null);

  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [excelPreview, setExcelPreview] = React.useState<ExcelPreview | null>(null);
  const [parsingExcel, setParsingExcel] = React.useState(false);

  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // 上传进度/日志面板（仿训练进度“控制台”）
  const [uploadConsoleOpen, setUploadConsoleOpen] = React.useState(false);
  const [uploadDoc, setUploadDoc] = React.useState<KnowledgeDocument | null>(null);
  const [uploadLogs, setUploadLogs] = React.useState<RagDocumentLog[]>([]);
  const [uploadPolling, setUploadPolling] = React.useState(false);
  const uploadPollTimerRef = React.useRef<number | null>(null);
  const uploadConsoleRef = React.useRef<HTMLDivElement | null>(null);

  const [createModalOpen, setCreateModalOpen] = React.useState(false);
  const [creatingCollection, setCreatingCollection] = React.useState(false);
  const [createForm] = Form.useForm();

  const selectedCollection = React.useMemo(
    () => collections.find((c) => c.id === collectionId) || null,
    [collections, collectionId]
  );

  const ensureToken = () => {
    if (!token) {
      message.warning('请先登录');
      return false;
    }
    return true;
  };

  const loadCollections = React.useCallback(async () => {
    if (!token) return;
    setCollectionsLoading(true);
    try {
      const list = await ragService.getCollections(token);
      setCollections(list);
      setCollectionId((prev) => prev ?? list[0]?.id ?? null);
    } catch (e: any) {
      message.error(e?.message || '加载知识库失败');
    } finally {
      setCollectionsLoading(false);
    }
  }, [token]);

  const loadDocuments = React.useCallback(async () => {
    if (!token || !collectionId) return;
    setDocumentsLoading(true);
    try {
      const docs = await ragService.getDocuments(collectionId, token);
      setDocuments(docs);
    } catch (e: any) {
      message.error(e?.message || '加载文档失败');
    } finally {
      setDocumentsLoading(false);
    }
  }, [token, collectionId]);

  const loadQueryHistory = React.useCallback(async () => {
    if (!token || !collectionId) return;
    setQueryHistoryLoading(true);
    try {
      const history = await ragService.getQueryHistory(collectionId, token, 10);
      setQueryHistory(history);
    } catch (e: any) {
      message.error(e?.message || '加载查询历史失败');
    } finally {
      setQueryHistoryLoading(false);
    }
  }, [token, collectionId]);

  React.useEffect(() => {
    if (!token) return;
    loadCollections();
  }, [token, loadCollections]);

  React.useEffect(() => {
    if (!token || !collectionId) {
      setDocuments([]);
      setQueryHistory([]);
      return;
    }
    loadDocuments();
    loadQueryHistory();
  }, [token, collectionId, loadDocuments, loadQueryHistory]);

  React.useEffect(() => {
    // 新日志到来后自动滚动到最底部
    const el = uploadConsoleRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [uploadLogs.length, uploadConsoleOpen]);

  React.useEffect(() => {
    return () => {
      if (uploadPollTimerRef.current) {
        window.clearInterval(uploadPollTimerRef.current);
        uploadPollTimerRef.current = null;
      }
    };
  }, []);

  const resetFile = () => {
    setSelectedFile(null);
    setExcelPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handlePickFile = async (file: File | null) => {
    if (!file) return;
    if (!isExcelFile(file)) {
      message.warning('当前仅支持上传 Excel（.xlsx/.xlsm）');
      resetFile();
      return;
    }
    setSelectedFile(file);
    setExcelPreview(null);
    setParsingExcel(true);
    try {
      const preview = await parseExcelPreview(file);
      setExcelPreview(preview);
      message.success('已读取 Excel 预览信息');
    } catch (e: any) {
      message.error(e?.message || '解析 Excel 失败');
      resetFile();
    } finally {
      setParsingExcel(false);
    }
  };

  const handleCreateCollection = async () => {
    if (!ensureToken()) return;
    setCreateModalOpen(true);
    createForm.setFieldsValue({
      name: '',
      description: ''
    });
  };

  const handleCreateOk = async () => {
    if (!ensureToken()) return;
    try {
      const values = await createForm.validateFields();
      setCreatingCollection(true);
      const created = await ragService.createCollection(values.name, values.description, token!);
      message.success('知识库创建成功');
      setCreateModalOpen(false);
      setCollections((prev) => [created, ...prev]);
      setCollectionId(created.id);
    } catch (e: any) {
      if (e?.errorFields) return; // 表单校验失败
      message.error(e?.message || '创建失败');
    } finally {
      setCreatingCollection(false);
    }
  };

  const handleQuery = async () => {
    if (!ensureToken()) return;
    if (!collectionId) {
      message.warning('请先选择知识库');
      return;
    }
    if (!queryText.trim()) {
      message.warning('请输入查询内容');
      return;
    }

    setIsQuerying(true);
    setAnswer('');
    setResultDocs([]);
    setQueryRuntime(null);
    try {
      const resp = await ragService.query(collectionId, queryText.trim(), token!);
      setAnswer(resp.response);
      setResultDocs(resp.documents || []);
      setQueryRuntime(resp.queryTime ?? null);
      await loadQueryHistory();
    } catch (e: any) {
      message.error(e?.message || '查询失败');
    } finally {
      setIsQuerying(false);
    }
  };

  const handleUpload = async () => {
    if (!ensureToken()) return;
    if (!collectionId) {
      message.warning('请先选择知识库');
      return;
    }
    if (!selectedFile || !excelPreview) {
      message.warning('请先选择并读取 Excel');
      return;
    }

    const startPollingUpload = (doc: KnowledgeDocument) => {
      if (!collectionId) return;
      if (uploadPollTimerRef.current) {
        window.clearInterval(uploadPollTimerRef.current);
        uploadPollTimerRef.current = null;
      }
      setUploadPolling(true);
      uploadPollTimerRef.current = window.setInterval(async () => {
        try {
          const [latestDoc, logs] = await Promise.all([
            ragService.getDocument(collectionId, doc.id, token!),
            ragService.getDocumentLogs(collectionId, doc.id, token!, 400)
          ]);
          setUploadDoc(latestDoc);
          setUploadLogs(logs);

          const status = String(latestDoc.uploadStatus || '').toLowerCase();
          if (status === 'completed' || status === 'failed') {
            if (uploadPollTimerRef.current) {
              window.clearInterval(uploadPollTimerRef.current);
              uploadPollTimerRef.current = null;
            }
            setUploadPolling(false);
            await loadCollections();
            await loadDocuments();
          }
        } catch (e) {
          // 轮询失败不打断用户
        }
      }, 1200);
    };

    const doUpload = async (overwrite: boolean) => {
      message.loading({ content: overwrite ? '上传中（覆盖）...' : '上传中...', key: 'ragUpload' });
      const doc = await ragService.uploadDocument(collectionId, selectedFile, token!, { overwrite });
      message.success({ content: '文件已接收，正在后台构建索引…', key: 'ragUpload' });
      setUploadDoc(doc);
      setUploadLogs([]);
      setUploadConsoleOpen(true);
      startPollingUpload(doc);
      resetFile();
    };

    Modal.confirm({
      title: '确认上传并构建索引？',
      content: (
        <Space direction="vertical" size={8}>
          <Text type="secondary">
            系统将使用「第一个 Sheet 的第一列」作为增强检索列（严格等值命中）。
          </Text>
          <Descriptions size="small" column={2} bordered>
            <Descriptions.Item label="文件">{excelPreview.filename}</Descriptions.Item>
            <Descriptions.Item label="大小">{formatBytes(excelPreview.fileSize)}</Descriptions.Item>
            <Descriptions.Item label="Sheet">{excelPreview.sheetName}</Descriptions.Item>
            <Descriptions.Item label="增强检索列">{excelPreview.primaryTagKey}</Descriptions.Item>
            <Descriptions.Item label="数据行数">{excelPreview.rowsTotal}</Descriptions.Item>
            <Descriptions.Item label="可索引行">{excelPreview.rowsIndexed}</Descriptions.Item>
            <Descriptions.Item label="空标签行">{excelPreview.rowsSkippedEmptyTag}</Descriptions.Item>
            <Descriptions.Item label="标签唯一数">{excelPreview.primaryTagUnique}</Descriptions.Item>
          </Descriptions>
        </Space>
      ),
      okText: '开始上传',
      cancelText: '取消',
      onOk: async () => {
        try {
          await doUpload(false);
        } catch (e: any) {
          const msg = e?.message || '上传失败';
          if (msg.includes('同名文档已存在') || msg.includes('确认覆盖')) {
            return new Promise<void>((resolve, reject) => {
              Modal.confirm({
                title: '发现同名文档',
                content: '该知识库中已存在同名 Excel。继续将覆盖并重建向量索引（历史索引会被删除）。是否确认覆盖？',
                okText: '确认覆盖',
                cancelText: '取消',
                onOk: async () => {
                  try {
                    await doUpload(true);
                    resolve();
                  } catch (err) {
                    message.error((err as any)?.message || '覆盖失败');
                    reject(err);
                  }
                },
                onCancel: () => resolve()
              });
            });
          }
          message.error({ content: msg, key: 'ragUpload' });
        }
      }
    });
  };

  const renderBoard = () => {
    const totalCollections = collections.length;
    const totalDocs = collections.reduce((sum, c) => sum + (c.documentCount || 0), 0);
    const selectedDocs = selectedCollection?.documentCount ?? 0;

    return (
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div>
          <Title level={3}>知识库看板</Title>
          <Text type="secondary">查看知识库规模、文档索引状态与最近查询。</Text>
        </div>

        <Row gutter={16}>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic title="知识库数量" value={totalCollections} />
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic title="文档总数" value={totalDocs} />
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic title="当前知识库文档" value={selectedDocs} />
            </Card>
          </Col>
        </Row>

        <Card title="当前知识库概览" extra={<Button onClick={loadCollections}>刷新</Button>}>
          {!selectedCollection ? (
            <Empty
              description={collectionsLoading ? '加载中...' : '暂无知识库，请先创建'}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            >
              <Button type="primary" onClick={handleCreateCollection} icon={<PlusOutlined />}>
                新建知识库
              </Button>
            </Empty>
          ) : (
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="名称">{selectedCollection.name}</Descriptions.Item>
              <Descriptions.Item label="文档数">{selectedCollection.documentCount}</Descriptions.Item>
              <Descriptions.Item label="Embedding 模型">{selectedCollection.embeddingModel}</Descriptions.Item>
              <Descriptions.Item label="更新时间">{dayjs(selectedCollection.updatedAt).format('YYYY-MM-DD HH:mm')}</Descriptions.Item>
              <Descriptions.Item label="描述" span={2}>
                {selectedCollection.description || '-'}
              </Descriptions.Item>
            </Descriptions>
          )}
        </Card>

        <Card title="最近查询" extra={<Button onClick={loadQueryHistory}>刷新</Button>}>
          {queryHistoryLoading ? (
            <Spin />
          ) : queryHistory.length === 0 ? (
            <Empty description="暂无查询历史" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          ) : (
            <Table
              rowKey="id"
              dataSource={queryHistory}
              pagination={false}
              columns={[
                { title: '问题', dataIndex: 'queryText' },
                { title: '命中数', dataIndex: 'resultCount', width: 90 },
                { title: '耗时(ms)', dataIndex: 'queryTimeMs', width: 110, render: (v) => v ?? '-' },
                {
                  title: '时间',
                  dataIndex: 'createdAt',
                  width: 180,
                  render: (v) => dayjs(v).format('YYYY-MM-DD HH:mm:ss')
                }
              ]}
            />
          )}
        </Card>
      </Space>
    );
  };

  const renderManage = () => {
    return (
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div>
          <Title level={3}>知识库管理</Title>
          <Text type="secondary">创建知识库、上传 Excel，并在上传前预览增强检索列与可索引行数。</Text>
        </div>

        <Card>
          <Space wrap size="middle" style={{ width: '100%', justifyContent: 'space-between' }}>
            <Space wrap size="middle">
              <div>
                <Text type="secondary">当前知识库</Text>
                <div>
                  <Select
                    style={{ width: 320 }}
                    placeholder="请选择知识库"
                    loading={collectionsLoading}
                    value={collectionId ?? undefined}
                    options={collections.map((c) => ({ value: c.id, label: `${c.name}（${c.documentCount}）` }))}
                    onChange={(v) => setCollectionId(v)}
                  />
                </div>
              </div>
              <Button icon={<PlusOutlined />} onClick={handleCreateCollection}>
                新建知识库
              </Button>
            </Space>

            {selectedCollection && (
              <Tag color="blue">
                文档 {selectedCollection.documentCount} · 更新时间 {dayjs(selectedCollection.updatedAt).format('MM-DD HH:mm')}
              </Tag>
            )}
          </Space>
        </Card>

        <Card title="上传 Excel（仅支持 .xlsx/.xlsm）" extra={<Text type="secondary">默认：第一个 Sheet + 第一列增强检索</Text>}>
          {!collectionId ? (
            <Empty description="请先选择或创建知识库" image={Empty.PRESENTED_IMAGE_SIMPLE}>
              <Button type="primary" onClick={handleCreateCollection} icon={<PlusOutlined />}>
                新建知识库
              </Button>
            </Empty>
          ) : (
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".xlsx,.xlsm"
                style={{ display: 'none' }}
                onChange={(e) => handlePickFile(e.target.files?.[0] || null)}
              />

              {!selectedFile ? (
                <Card
                  style={{ borderStyle: 'dashed' }}
                  bodyStyle={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}
                >
                  <Space direction="vertical" size={4}>
                    <Text strong>选择 Excel 文件</Text>
                    <Text type="secondary">上传前会自动读取：增强检索列、可索引行数、唯一标签数等信息。</Text>
                  </Space>
                  <Button type="primary" icon={<UploadOutlined />} onClick={() => fileInputRef.current?.click()}>
                    选择文档
                  </Button>
                </Card>
              ) : (
                <Card
                  style={{ borderStyle: 'dashed' }}
                  bodyStyle={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}
                >
                  <Space direction="vertical" size={12} style={{ flex: 1 }}>
                    <Space wrap>
                      <Tag color="geekblue">{selectedFile.name}</Tag>
                      <Tag>{formatBytes(selectedFile.size)}</Tag>
                      {parsingExcel ? <Tag color="gold">解析中...</Tag> : excelPreview ? <Tag color="green">已解析</Tag> : null}
                    </Space>

                    {excelPreview ? (
                      <Descriptions bordered size="small" column={2}>
                        <Descriptions.Item label="Sheet">{excelPreview.sheetName}</Descriptions.Item>
                        <Descriptions.Item label="增强检索列">{excelPreview.primaryTagKey}</Descriptions.Item>
                        <Descriptions.Item label="数据行数">{excelPreview.rowsTotal}</Descriptions.Item>
                        <Descriptions.Item label="可索引行">{excelPreview.rowsIndexed}</Descriptions.Item>
                        <Descriptions.Item label="空标签行">{excelPreview.rowsSkippedEmptyTag}</Descriptions.Item>
                        <Descriptions.Item label="标签唯一数">{excelPreview.primaryTagUnique}</Descriptions.Item>
                      </Descriptions>
                    ) : (
                      <Text type="secondary">请等待解析完成（或文件内容为空/格式异常）。</Text>
                    )}
                  </Space>

                  <Space direction="vertical">
                    <Button onClick={() => fileInputRef.current?.click()}>更换文件</Button>
                    <Button type="primary" onClick={handleUpload} disabled={!excelPreview} loading={parsingExcel}>
                      上传文档并生成 RAG
                    </Button>
                    <Button danger onClick={resetFile}>
                      清空
                    </Button>
                  </Space>
                </Card>
              )}
            </Space>
          )}
        </Card>

        <Card title={`已索引文档（${documents.length}）`} extra={<Button onClick={loadDocuments}>刷新</Button>}>
          {documentsLoading ? (
            <Spin />
          ) : documents.length === 0 ? (
            <Empty description="暂无文档" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          ) : (
            <Table
              rowKey="id"
              dataSource={documents}
              pagination={{ pageSize: 8 }}
              columns={[
                { title: '文件名', dataIndex: 'filename' },
                { title: '行数(Chunk)', dataIndex: 'chunkCount', width: 110 },
                { title: '大小', dataIndex: 'fileSize', width: 120, render: (v) => formatBytes(v) },
                {
                  title: '状态',
                  dataIndex: 'uploadStatus',
                  width: 110,
                  render: (v: string) => (
                    <Tag color={v === 'completed' ? 'green' : v === 'processing' ? 'blue' : 'default'}>{v}</Tag>
                  )
                },
                {
                  title: '时间',
                  dataIndex: 'createdAt',
                  width: 180,
                  render: (v) => dayjs(v).format('YYYY-MM-DD HH:mm')
                }
              ]}
            />
          )}
        </Card>
      </Space>
    );
  };

  const renderQuery = () => {
    return (
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <div>
          <Title level={3}>知识库查询</Title>
          <Text type="secondary">输入问题，返回规则化摘要 + 证据链（命中行）。</Text>
        </div>

        <Card>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Space wrap size="middle" style={{ justifyContent: 'space-between', width: '100%' }}>
              <Select
                style={{ width: 340 }}
                placeholder="请选择知识库"
                loading={collectionsLoading}
                value={collectionId ?? undefined}
                options={collections.map((c) => ({ value: c.id, label: `${c.name}（${c.documentCount}）` }))}
                onChange={(v) => setCollectionId(v)}
              />
              <Button onClick={() => loadQueryHistory()}>刷新历史</Button>
            </Space>

            <Space.Compact style={{ width: '100%' }}>
              <Input
                placeholder="例如：停充码1001代表什么含义？"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onPressEnter={handleQuery}
              />
              <Button type="primary" icon={<SearchOutlined />} loading={isQuerying} onClick={handleQuery}>
                查询
              </Button>
            </Space.Compact>

            {answer ? (
              <Card size="small" title="查询结果">
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                  <Space wrap>
                    <Tag color="blue">命中 {resultDocs.length}</Tag>
                    {queryRuntime !== null && <Tag>耗时 {queryRuntime} ms</Tag>}
                  </Space>
                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.8 }}>{answer}</div>
                </Space>
              </Card>
            ) : (
              <Empty description="暂无结果" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            )}

            <Card size="small" title="证据链（命中行）">
              {resultDocs.length === 0 ? (
                <Empty description="暂无命中" image={Empty.PRESENTED_IMAGE_SIMPLE} />
              ) : (
                <Table
                  rowKey={(r, idx) => String(r?.id ?? idx)}
                  dataSource={resultDocs}
                  pagination={{ pageSize: 5 }}
                  columns={[
                    { title: '来源文件', dataIndex: 'filename', width: 260, render: (v) => v || '-' },
                    { title: 'Sheet', dataIndex: 'sheet_name', width: 120, render: (_: any, r: any) => r?.sheet_name ?? r?.sheetName ?? '-' },
                    { title: '行号', dataIndex: 'row_index', width: 90, render: (_: any, r: any) => r?.row_index ?? r?.rowIndex ?? '-' },
                    {
                      title: '摘要',
                      dataIndex: 'snippet',
                      render: (_: any, r: any) => r?.snippet || r?.content || '-'
                    }
                  ]}
                />
              )}
            </Card>
          </Space>
        </Card>

        <Card title="查询历史">
          {queryHistoryLoading ? (
            <Spin />
          ) : queryHistory.length === 0 ? (
            <Empty description="暂无查询历史" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          ) : (
            <Table
              rowKey="id"
              dataSource={queryHistory}
              pagination={false}
              onRow={(record) => ({
                onClick: () => setQueryText(record.queryText)
              })}
              columns={[
                { title: '问题', dataIndex: 'queryText' },
                { title: '命中数', dataIndex: 'resultCount', width: 90 },
                { title: '耗时(ms)', dataIndex: 'queryTimeMs', width: 110, render: (v) => v ?? '-' },
                {
                  title: '时间',
                  dataIndex: 'createdAt',
                  width: 180,
                  render: (v) => dayjs(v).format('YYYY-MM-DD HH:mm:ss')
                }
              ]}
            />
          )}
        </Card>
      </Space>
    );
  };

  const renderContent = () => {
    switch (activeMenu) {
      case 'board':
        return renderBoard();
      case 'manage':
        return renderManage();
      case 'query':
        return renderQuery();
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
              RAG 控制台
            </Title>
            <Text type="secondary">知识库看板 / 管理 / 查询</Text>
          </div>
          <Menu
            mode="inline"
            selectedKeys={[activeMenu]}
            items={menuItems}
            onClick={(info) => setActiveMenu(info.key as MenuKey)}
            style={{ borderRight: 'none' }}
          />
        </Sider>
        <Content>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            {renderContent()}
          </Space>
        </Content>
      </Layout>

      <Modal
        title="新建知识库"
        open={createModalOpen}
        onOk={handleCreateOk}
        onCancel={() => setCreateModalOpen(false)}
        okText="创建"
        cancelText="取消"
        confirmLoading={creatingCollection}
      >
        <Form layout="vertical" form={createForm}>
          <Form.Item label="知识库名称" name="name" rules={[{ required: true, message: '请输入知识库名称' }]}>
            <Input placeholder="例如：充电故障提示知识库" maxLength={100} />
          </Form.Item>
          <Form.Item label="描述（可选）" name="description">
            <Input.TextArea rows={3} placeholder="简单描述该知识库用途" maxLength={2000} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="文档上传与索引进度"
        open={uploadConsoleOpen}
        onCancel={() => {
          setUploadConsoleOpen(false);
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              setUploadConsoleOpen(false);
            }}
          >
            关闭
          </Button>
        ]}
        width={860}
      >
        {!uploadDoc ? (
          <Empty description="暂无任务" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        ) : (
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="文件">{uploadDoc.filename}</Descriptions.Item>
              <Descriptions.Item label="文档ID">{uploadDoc.id}</Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag
                  color={
                    uploadDoc.uploadStatus === 'completed'
                      ? 'green'
                      : uploadDoc.uploadStatus === 'failed'
                        ? 'red'
                        : 'blue'
                  }
                >
                  {uploadDoc.uploadStatus}
                </Tag>
                {uploadPolling && <Tag color="gold">轮询中</Tag>}
              </Descriptions.Item>
              <Descriptions.Item label="Chunk">
                {typeof uploadDoc.chunkCount === 'number' ? uploadDoc.chunkCount : '-'}
              </Descriptions.Item>
              {uploadDoc.processingError && (
                <Descriptions.Item label="错误" span={2}>
                  <Text type="danger">{uploadDoc.processingError}</Text>
                </Descriptions.Item>
              )}
            </Descriptions>

            <Card size="small" title="处理日志（实时刷新）">
              {uploadLogs.length === 0 ? (
                <Empty description="暂无日志" image={Empty.PRESENTED_IMAGE_SIMPLE} />
              ) : (
                <div className="console-log" ref={uploadConsoleRef}>
                  {uploadLogs.map((item) => {
                    const level = String(item.logLevel || '').toLowerCase();
                    const levelClass =
                      level.includes('error') || level.includes('critical')
                        ? 'level-error'
                        : level.includes('warn')
                          ? 'level-warn'
                          : level.includes('debug')
                            ? 'level-debug'
                            : 'level-info';
                    return (
                      <div key={item.id} className="console-log-line">
                        <span className="console-log-time">{dayjs(item.createdAt).format('HH:mm:ss')}</span>
                        <span className={`console-log-level ${levelClass}`}>{item.logLevel}</span>
                        <span className="console-log-message">{item.message}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </Card>
          </Space>
        )}
      </Modal>
    </Card>
  );
};

export default RagCenter;

