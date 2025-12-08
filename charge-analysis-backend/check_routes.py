#!/usr/bin/env python3
"""检查所有注册的路由"""
from main import app

print("=" * 60)
print("所有注册的路由:")
print("=" * 60)

analyses_routes = []
for route in app.routes:
    if hasattr(route, 'path') and hasattr(route, 'methods'):
        methods = list(route.methods) if route.methods else []
        path = route.path
        if '/api/analyses' in path:
            analyses_routes.append((methods, path))

for methods, path in sorted(analyses_routes):
    marker = "✅" if "cancel" in path else "  "
    print(f"{marker} {methods[0] if methods else 'GET':8} {path}")

print("=" * 60)
if any("cancel" in path for _, path in analyses_routes):
    print("✅ cancel 路由已注册")
else:
    print("❌ cancel 路由未找到！")
