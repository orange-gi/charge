const DEFAULT_BASE_URL = 'http://localhost:8000';

export const API_BASE_URL = ((import.meta as any).env?.VITE_API_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, '');

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
