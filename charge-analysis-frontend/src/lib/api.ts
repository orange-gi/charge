function inferDefaultBaseUrl(): string {
  // 部署到远程时，前端在浏览器里运行：localhost 会指向“访问者本机”，而不是服务器。
  // 因此默认用“当前页面的 hostname + :8000”去访问同机后端。
  // 如需自定义，使用环境变量：VITE_API_BASE_URL
  if (typeof window !== 'undefined' && window.location) {
    const { protocol, hostname } = window.location;
    if (hostname && hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return `${protocol}//${hostname}:8000`;
    }
  }
  return 'http://localhost:8000';
}

const DEFAULT_BASE_URL = inferDefaultBaseUrl();

export const API_BASE_URL = String(((import.meta as any).env?.VITE_API_BASE_URL || DEFAULT_BASE_URL)).replace(/\/$/, '');

interface RequestOptions {
  method?: string;
  token?: string | null;
  body?: any;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

export async function apiRequest<T = unknown>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', token, body, headers = {}, signal } = options;
  const url = `${API_BASE_URL}${path}`;
  const requestHeaders: Record<string, string> = { ...headers };

  if (token) {
    requestHeaders['Authorization'] = `Bearer ${token}`;
  }

  let payload: BodyInit | undefined;
  if (body instanceof FormData) {
    payload = body;
  } else if (body !== undefined && body !== null) {
    requestHeaders['Content-Type'] = requestHeaders['Content-Type'] || 'application/json';
    payload = JSON.stringify(body);
  }

  const response = await fetch(url, {
    method,
    headers: requestHeaders,
    body: payload,
    signal
  });

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const message = data?.detail || data?.message || data?.error || '请求失败';
    throw new Error(message);
  }

  return data as T;
}
