#!/usr/bin/env python3
"""测试所有API路由是否正确注册"""
from main import app

print("=" * 70)
print("📋 所有注册的 API 路由:")
print("=" * 70)

# 按路径分组
routes_by_prefix = {}
for route in app.routes:
    if hasattr(route, 'path') and hasattr(route, 'methods'):
        methods = list(route.methods) if route.methods else ['GET']
        path = route.path
        
        # 提取前缀
        prefix = '/api/analyses'
        if path.startswith(prefix):
            if prefix not in routes_by_prefix:
                routes_by_prefix[prefix] = []
            routes_by_prefix[prefix].append((methods[0], path))

# 显示 analyses 相关的路由
if '/api/analyses' in routes_by_prefix:
    print(f"\n🔍 {routes_by_prefix['/api/analyses'][0][1].split('/')[2] if routes_by_prefix['/api/analyses'] else ''} 相关路由:")
    for method, path in sorted(routes_by_prefix['/api/analyses']):
        marker = "✅" if "cancel" in path else "  "
        print(f"  {marker} {method:6} {path}")

print("\n" + "=" * 70)
print("💡 如果 cancel 路由存在，但访问时404，请重启后端服务:")
print("   1. 停止当前服务 (Ctrl+C)")
print("   2. 重新运行: uvicorn main:app --reload --host 127.0.0.1 --port 8000")
print("=" * 70)

