from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Optional


_RE_INVISIBLE_SPACE = re.compile(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]")
_RE_WHITESPACE_RUN = re.compile(r"\s+")
_RE_NUMBER_TOKEN = re.compile(r"\d+")
_RE_KV_NUMBER = re.compile(r"(?P<key>[^=]{1,80})=(?P<val>\d+)")


def normalize_text(text: str) -> str:
    """通用文本规范化：NFKC + 空白规整。"""
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = _RE_INVISIBLE_SPACE.sub(" ", s)
    s = _RE_WHITESPACE_RUN.sub(" ", s).strip()
    return s


def normalize_primary_tag_value(value: object) -> str:
    """主标签值规范化（用于严格等值匹配）。

    目标：解决 Excel 数字单元格、全角数字、尾随 .0、科学计数法等导致的漏命中。
    """
    if value is None:
        return ""

    # bool 是 int 的子类，需先排除
    if isinstance(value, bool):
        return "1" if value else "0"

    # 数字（int/float/Decimal）优先走数值路径
    if isinstance(value, (int, Decimal)):
        return str(int(value))
    if isinstance(value, float):
        # 规避 NaN/Inf
        if value != value or value in (float("inf"), float("-inf")):
            return ""
        # float 1001.0 -> 1001
        if value.is_integer():
            return str(int(value))
        # 非整数：避免科学计数法带来的字符串不稳定
        return format(value, "f").rstrip("0").rstrip(".")

    s = normalize_text(str(value))
    if not s:
        return ""

    # 处理类似 "1001.0" / "1.001E3"
    try:
        d = Decimal(s)
        # 若是整数（含 1001.0 / 1E3），转为 int 字符串
        if d == d.to_integral_value():
            return str(int(d))
        # 否则保留去尾零的普通十进制形式
        normalized = format(d.normalize(), "f")
        normalized = normalized.rstrip("0").rstrip(".")
        return normalized
    except (InvalidOperation, ValueError):
        # 非纯数：直接返回 NFKC + 空白规整后的字符串
        return s


def extract_candidate_primary_tag(query: str) -> Optional[str]:
    """从用户 query 中提取可能的主标签值（停充码/错误码等）。

    当前策略：提取第一个连续数字串（NFKC 后）。
    """
    s = normalize_text(query)
    m = _RE_NUMBER_TOKEN.search(s)
    if not m:
        return None
    return normalize_primary_tag_value(m.group(0))


def extract_candidate_primary_tag_kv(query: str) -> Optional[tuple[str, str]]:
    """从 query 中提取“主标签键=数字值”。

    用途：严格等值命中时同时约束 primary_tag_key + primary_tag_value，
    避免出现“BMS_DCChrgSt=2 被误命中为 停充码ChrgEndNum 8bit=2”的跨表错配。
    """
    s = normalize_text(query)
    m = _RE_KV_NUMBER.search(s)
    if not m:
        return None
    key = normalize_text(m.group("key"))
    val = normalize_primary_tag_value(m.group("val"))
    if not key or not val:
        return None
    return key, val

