// Supabase Configuration
export const SUPABASE_URL = 'https://ahmzlbndtclnbiptpvex.supabase.co';
export const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFobXpsYm5kdGNsbmJpcHRwdmV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1MzM3MzIsImV4cCI6MjA3OTEwOTczMn0.crigFt3xKl88S9YLlfqUTUGlyeE7dPC-3u6XTKOmdmQ';

// Edge Function URLs
export const EDGE_FUNCTIONS = {
  userAuth: `${SUPABASE_URL}/functions/v1/user-auth`,
  fileUpload: `${SUPABASE_URL}/functions/v1/file-upload`,
  chargingAnalysis: `${SUPABASE_URL}/functions/v1/charging-analysis-v2`,
  ragQuery: `${SUPABASE_URL}/functions/v1/rag-query`,
  trainingManagement: `${SUPABASE_URL}/functions/v1/training-management`
};

// Supabase client helper
export async function callEdgeFunction(
  functionName: keyof typeof EDGE_FUNCTIONS,
  body: any,
  token?: string
) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'apikey': SUPABASE_ANON_KEY,
    'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
  };

  // If user token is provided, pass it via custom header
  if (token) {
    headers['x-custom-token'] = token;
  }

  const response = await fetch(EDGE_FUNCTIONS[functionName], {
    method: 'POST',
    headers,
    body: JSON.stringify(body)
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    throw new Error(data.error?.message || 'Request failed');
  }

  return data;
}

// Direct Supabase REST API helper
export async function querySupabase(
  table: string,
  method: 'GET' | 'POST' | 'PATCH' | 'DELETE' = 'GET',
  options: {
    select?: string;
    filter?: Record<string, any>;
    body?: any;
    order?: { column: string; ascending?: boolean };
    limit?: number;
    single?: boolean;
  } = {}
) {
  let url = `${SUPABASE_URL}/rest/v1/${table}`;

  // Add select
  if (options.select) {
    url += `?select=${options.select}`;
  }

  // Add filters
  if (options.filter) {
    const hasQuery = url.includes('?');
    Object.entries(options.filter).forEach(([key, value], index) => {
      const separator = hasQuery || index > 0 ? '&' : '?';
      url += `${separator}${key}=eq.${encodeURIComponent(value)}`;
    });
  }

  // Add order
  if (options.order) {
    url += url.includes('?') ? '&' : '?';
    url += `order=${options.order.column}.${options.order.ascending === false ? 'desc' : 'asc'}`;
  }

  // Add limit
  if (options.limit) {
    url += url.includes('?') ? '&' : '?';
    url += `limit=${options.limit}`;
  }

  const headers: Record<string, string> = {
    'apikey': SUPABASE_ANON_KEY,
    'Content-Type': 'application/json'
  };

  if (method !== 'GET' && options.body) {
    headers['Prefer'] = 'return=representation';
  }

  const fetchOptions: RequestInit = {
    method,
    headers
  };

  if (options.body && method !== 'GET') {
    fetchOptions.body = JSON.stringify(options.body);
  }

  const response = await fetch(url, fetchOptions);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.message || 'Database query failed');
  }

  // Return single item if requested
  if (options.single) {
    return data[0] || null;
  }

  return data;
}
