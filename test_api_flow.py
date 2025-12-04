import os
import time
from pathlib import Path
from uuid import uuid4

import requests

API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000').rstrip('/')
TEST_FILE = Path('/workspace/test_data/test-charging-correct-format.csv')


def auth_headers(token: str) -> dict:
  return {
      'Authorization': f'Bearer {token}'
  }


def register_or_login(email: str, password: str) -> tuple[str, dict]:
  username = f'user_{uuid4().hex[:8]}'
  register_payload = {
      'email': email,
      'password': password,
      'username': username,
      'first_name': 'Test',
      'last_name': 'User'
  }

  register_resp = requests.post(f'{API_BASE_URL}/api/auth/register', json=register_payload, timeout=15)
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
  with TEST_FILE.open('rb') as file_obj:
    files = {'file': (TEST_FILE.name, file_obj, 'text/csv')}
    data = {'analysis_name': 'API Flow 测试', 'description': '自动化脚本上传'}
    response = requests.post(
        f'{API_BASE_URL}/api/analyses/upload',
        headers=auth_headers(token),
        files=files,
        data=data,
        timeout=30
    )
  if response.status_code != 200:
    raise RuntimeError(f'文件上传失败: {response.text}')
  analysis = response.json()
  print(f"✅ 创建分析：ID={analysis['id']}")
  return analysis


def start_analysis(token: str, analysis_id: int) -> None:
  response = requests.post(
      f'{API_BASE_URL}/api/analyses/{analysis_id}/run',
      headers=auth_headers(token),
      timeout=15
  )
  if response.status_code != 200:
    raise RuntimeError(f'启动分析失败: {response.text}')
  print('🚀 分析任务已启动')


def wait_for_completion(token: str, analysis_id: int, timeout: int = 120) -> dict:
  deadline = time.time() + timeout
  while time.time() < deadline:
    response = requests.get(
        f'{API_BASE_URL}/api/analyses/{analysis_id}',
        headers=auth_headers(token),
        timeout=15
    )
    if response.status_code != 200:
      raise RuntimeError(f'查询分析状态失败: {response.text}')
    analysis = response.json()
    status = analysis['status']
    print(f'📊 当前状态: {status}({analysis.get("progress", 0)}%)')
    if status in {'completed', 'failed'}:
      return analysis
    time.sleep(5)

  raise TimeoutError('等待分析完成超时')


def fetch_results(token: str, analysis_id: int) -> list[dict]:
  response = requests.get(
      f'{API_BASE_URL}/api/analyses/{analysis_id}/results',
      headers=auth_headers(token),
      timeout=15
  )
  if response.status_code != 200:
    raise RuntimeError(f'获取分析结果失败: {response.text}')
  payload = response.json()
  return payload.get('results', [])


def main() -> None:
  email = f"test_user_{int(time.time())}@example.com"
  password = 'TestPass123!'

  print('=== 用户注册 / 登录 ===')
  token, user = register_or_login(email, password)
  print(f"当前用户: {user['email']}")

  print('\n=== 上传充电数据文件 ===')
  analysis = upload_analysis(token)

  print('\n=== 启动分析 ===')
  start_analysis(token, analysis['id'])

  print('\n=== 等待分析完成 ===')
  completed_analysis = wait_for_completion(token, analysis['id'])
  print(f"分析完成，状态: {completed_analysis['status']}, 进度: {completed_analysis.get('progress')}%")

  print('\n=== 获取分析结果 ===')
  results = fetch_results(token, analysis['id'])
  print(f'共获得 {len(results)} 条结果')
  for item in results:
    print(f" - [{item.get('resultType', item.get('type'))}] {item.get('title')}")

  print('\n流程执行完毕 ✅')


if __name__ == '__main__':
  main()
