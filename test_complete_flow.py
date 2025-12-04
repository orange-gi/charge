import os
import time
from pathlib import Path
from uuid import uuid4

import requests

API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000').rstrip('/')
TEST_ANALYSIS_FILE = Path('/workspace/test_data/test-charging-correct-format.csv')

TRAINING_DATASET = """feature1,feature2,feature3,label
1.2,3.4,5.6,0
2.3,4.5,6.7,1
3.4,5.6,7.8,0
4.5,6.7,8.9,1
5.6,7.8,9.0,0
""".strip()


def auth_headers(token: str) -> dict:
  return {'Authorization': f'Bearer {token}'}


def register_or_login(email: str, password: str) -> tuple[str, dict]:
  username = f'user_{uuid4().hex[:8]}'
  payload = {
      'email': email,
      'password': password,
      'username': username,
      'first_name': 'Auto',
      'last_name': 'Tester'
  }
  register_resp = requests.post(f'{API_BASE_URL}/api/auth/register', json=payload, timeout=15)
  if register_resp.status_code == 200:
    data = register_resp.json()
    print('✅ 注册成功')
    return data['token'], data['user']
  if register_resp.status_code != 400:
    raise RuntimeError(f'注册失败: {register_resp.text}')

  login_resp = requests.post(
      f'{API_BASE_URL}/api/auth/login',
      json={'email': email, 'password': password},
      timeout=15
  )
  if login_resp.status_code != 200:
    raise RuntimeError(f'登录失败: {login_resp.text}')
  data = login_resp.json()
  print('✅ 登录成功')
  return data['token'], data['user']


def upload_analysis(token: str) -> dict:
  with TEST_ANALYSIS_FILE.open('rb') as fp:
    files = {'file': (TEST_ANALYSIS_FILE.name, fp, 'text/csv')}
    data = {'analysis_name': '完整流程测试', 'description': '后端API全流程'}
    response = requests.post(
        f'{API_BASE_URL}/api/analyses/upload',
        headers=auth_headers(token),
        files=files,
        data=data,
        timeout=30
    )
  if response.status_code != 200:
    raise RuntimeError(f'上传失败: {response.text}')
  analysis = response.json()
  print(f"✅ 已创建分析（ID={analysis['id']}）")
  return analysis


def start_analysis(token: str, analysis_id: int) -> None:
  resp = requests.post(
      f'{API_BASE_URL}/api/analyses/{analysis_id}/run',
      headers=auth_headers(token),
      timeout=15
  )
  if resp.status_code != 200:
    raise RuntimeError(f'启动分析失败: {resp.text}')
  print('🚀 分析已启动')


def wait_for_analysis(token: str, analysis_id: int, timeout: int = 180) -> None:
  deadline = time.time() + timeout
  while time.time() < deadline:
    resp = requests.get(
        f'{API_BASE_URL}/api/analyses/{analysis_id}',
        headers=auth_headers(token),
        timeout=15
    )
    if resp.status_code != 200:
      raise RuntimeError(f'查询分析失败: {resp.text}')
    analysis = resp.json()
    status = analysis['status']
    print(f"📊 分析状态: {status} ({analysis.get('progress', 0)}%)")
    if status in {'completed', 'failed'}:
      print('✅ 分析流程结束')
      return
    time.sleep(5)
  raise TimeoutError('分析等待超时')


def upload_training_dataset(token: str) -> int:
  files = {'file': ('training-data.csv', TRAINING_DATASET.encode('utf-8'), 'text/csv')}
  data = {'name': '自动化训练集', 'description': 'API 集成测试', 'dataset_type': 'standard'}
  resp = requests.post(
      f'{API_BASE_URL}/api/training/datasets',
      headers=auth_headers(token),
      files=files,
      data=data,
      timeout=30
  )
  if resp.status_code != 200:
    raise RuntimeError(f'上传训练集失败: {resp.text}')
  payload = resp.json()
  dataset_id = payload['dataset_id']
  print(f'✅ 训练数据集已创建(ID={dataset_id})')
  return dataset_id


def create_training_task(token: str, dataset_id: int) -> int:
  resp = requests.post(
      f'{API_BASE_URL}/api/training/tasks',
      headers=auth_headers(token),
      json={
          'name': '自动化训练任务',
          'description': '脚本触发',
          'dataset_id': dataset_id,
          'model_type': 'flow_control',
          'hyperparameters': {'epochs': 3, 'batch_size': 8}
      },
      timeout=15
  )
  if resp.status_code != 200:
    raise RuntimeError(f'创建任务失败: {resp.text}')
  payload = resp.json()
  task_id = payload['task_id']
  print(f'✅ 训练任务已创建(ID={task_id})')
  return task_id


def start_training(token: str, task_id: int) -> None:
  resp = requests.post(
      f'{API_BASE_URL}/api/training/tasks/{task_id}/start',
      headers=auth_headers(token),
      timeout=15
  )
  if resp.status_code != 200:
    raise RuntimeError(f'启动训练失败: {resp.text}')
  print('🚀 训练任务已启动')


def wait_for_training(token: str, task_id: int, timeout: int = 180) -> None:
  deadline = time.time() + timeout
  while time.time() < deadline:
    resp = requests.get(
        f'{API_BASE_URL}/api/training/tasks/{task_id}',
        headers=auth_headers(token),
        timeout=15
    )
    if resp.status_code != 200:
      raise RuntimeError(f'查询训练状态失败: {resp.text}')
    task = resp.json()
    status = task['status']
    print(f"⚙️ 训练状态: {status} (进度 {task.get('progress', 0)}%)")
    if status in {'completed', 'failed'}:
      print('✅ 训练流程结束')
      return
    time.sleep(5)
  raise TimeoutError('训练等待超时')


def main() -> None:
  email = f"flow_user_{int(time.time())}@example.com"
  password = 'TestPass123!'

  print('=== 用户认证 ===')
  token, user = register_or_login(email, password)
  print(f"当前用户: {user['email']}")

  print('\n=== 上传并运行充电分析 ===')
  analysis = upload_analysis(token)
  start_analysis(token, analysis['id'])
  wait_for_analysis(token, analysis['id'])

  print('\n=== 训练管理流程 ===')
  dataset_id = upload_training_dataset(token)
  task_id = create_training_task(token, dataset_id)
  start_training(token, task_id)
  wait_for_training(token, task_id)

  print('\n所有流程执行完毕 ✅')


if __name__ == '__main__':
  main()
