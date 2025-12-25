"""
独立验证：OpenAI 兼容接口（内网 LLM）连通性 / 代理影响 / 超时。

用法示例：
  1) 直接用 .env 配置（推荐）：
     python3 verify_llm_connectivity.py

  2) 临时指定（会覆盖 .env / 环境变量）：
     OPENAI_BASE_URL="http://api.openai.rnd.huawei.com/v1" \
     OPENAI_API_KEY="sk-1234" \
     LLM_MODEL_NAME="deepseek-r1-distill-qwen-32b" \
     NO_PROXY="127.0.0.1,::1,localhost,api.openai.rnd.huawei.com" \
     python3 verify_llm_connectivity.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


def _print_env(key: str) -> None:
    v = os.environ.get(key)
    print(f"{key}={v!r}")


def _ensure_no_proxy_for_host(base_url: str) -> None:
    """确保 NO_PROXY/no_proxy 至少包含 base_url 的 host（不带端口）。"""
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if not host:
        return
    cur = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [p.strip() for p in cur.split(",") if p.strip()]
    if host not in parts:
        parts.append(host)
    merged = ",".join(parts)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _load_settings() -> tuple[str, str, str]:
    """
    优先级：
    1) 先尝试加载同目录 .env 到 os.environ（不依赖 pydantic）
    2) 再尝试走后端 config.get_settings（若依赖已安装）
    3) 最后直接读环境变量
    """
    backend_dir = Path(__file__).resolve().parent
    env_path = backend_dir / ".env"

    # 1) 尝试 python-dotenv
    if env_path.exists():
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(dotenv_path=str(env_path), override=False)
        except Exception:
            # 兜底：最简解析 KEY=VALUE
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
            except Exception:
                pass

    # 2) 尝试后端 Settings（可选）
    try:
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        from config import get_settings  # type: ignore

        s = get_settings()
        base_url = str(getattr(s, "openai_base_url", "") or "").rstrip("/")
        api_key = str(getattr(s, "openai_api_key", "") or "")
        model = str(getattr(s, "llm_model_name", "") or "")
        return base_url, api_key, model
    except Exception as exc:
        # 打印一次原因，便于定位“为什么读不到 .env / config”
        print("NOTE: 读取后端 config.get_settings 失败，将回退到环境变量：", repr(exc))

    # 3) 最后直接走环境变量
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    model = os.environ.get("LLM_MODEL_NAME") or os.environ.get("MODEL") or ""
    return base_url, api_key, model


def _test_sync(base_url: str, api_key: str, model: str, stream: bool) -> None:
    from openai import OpenAI

    print(f"\n== Sync OpenAI test (stream={stream}) ==")
    t0 = time.time()
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=0)
    try:
        if stream:
            stream_resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "你是什么模型？请只回答模型名。"}],
                stream=True,
            )
            out = []
            for chunk in stream_resp:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    out.append(delta.content)
            print("response:", "".join(out).strip())
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "你是什么模型？请只回答模型名。"}],
                stream=False,
            )
            print("response:", (resp.choices[0].message.content or "").strip())
        print(f"elapsed_ms={int((time.time() - t0) * 1000)}")
    except Exception as exc:
        print("ERROR(sync):", repr(exc))


async def _test_async(base_url: str, api_key: str, model: str, trust_env: bool) -> None:
    from openai import AsyncOpenAI

    print(f"\n== AsyncOpenAI test (trust_env={trust_env}) ==")
    t0 = time.time()
    # 注意：不同 openai SDK 版本对 http_client 类型要求不同。
    # 这里使用 httpx.AsyncClient（最兼容的写法）。
    try:
        import httpx  # type: ignore

        http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0), trust_env=trust_env)
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=180.0,
            max_retries=0,
            http_client=http_client,
        )
    except Exception:
        # 兜底：不注入自定义 http_client，只测基本连通性
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=0)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "你是什么模型？请只回答模型名。"}],
            stream=False,
        )
        print("response:", (resp.choices[0].message.content or "").strip())
        print(f"elapsed_ms={int((time.time() - t0) * 1000)}")
    except Exception as exc:
        print("ERROR(async):", repr(exc))
    finally:
        # 如果注入了 httpx.AsyncClient，记得关闭，避免资源泄露
        try:
            await client.close()
        except Exception:
            pass


def main() -> int:
    base_url, api_key, model = _load_settings()
    if not base_url:
        print("ERROR: OPENAI_BASE_URL 为空（请在 .env 或环境变量里设置）")
        return 2
    if not api_key:
        print("WARNING: OPENAI_API_KEY 为空（若内网 LLM 不校验可忽略）")
    if not model:
        print("ERROR: LLM_MODEL_NAME 为空（请在 .env 或环境变量里设置）")
        return 2

    _ensure_no_proxy_for_host(base_url)

    print("== Effective config ==")
    print("base_url=", base_url)
    print("model=", model)
    _print_env("HTTP_PROXY")
    _print_env("HTTPS_PROXY")
    _print_env("NO_PROXY")
    _print_env("no_proxy")

    # 1) 同步：stream/非stream 都测一次
    _test_sync(base_url, api_key, model, stream=True)
    _test_sync(base_url, api_key, model, stream=False)

    # 2) 异步：分别测 trust_env=True/False（定位是不是环境代理导致的 504）
    import asyncio

    asyncio.run(_test_async(base_url, api_key, model, trust_env=True))
    asyncio.run(_test_async(base_url, api_key, model, trust_env=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

