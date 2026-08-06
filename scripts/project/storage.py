"""项目JSON的原子读写、安全路径解析和名称校验。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json_object(path: str | Path, *, description: str = "JSON 配置") -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"缺少{description}: {target}")
    data = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{description}根节点必须是对象: {target}")
    return data


def write_json_atomic(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def write_text_atomic(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_safe_name(value: str, field: str, *, allow_dot: bool = False) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if allow_dot:
        allowed += "."
    if not value or any(char not in allowed for char in value):
        extra = "、点" if allow_dot else ""
        raise ValueError(f"{field} 只能包含字母、数字{extra}、横线和下划线")


def resolve_within(root: str | Path, value: str | Path, *, field: str = "path") -> Path:
    """解析项目内部路径并拒绝绝对路径或 .. 越界。"""

    project_root = Path(root).resolve()
    raw_path = Path(value)
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
    else:
        candidate = (project_root / raw_path).resolve()
    if not candidate.is_relative_to(project_root):
        raise ValueError(f"{field} 必须位于 SOP 项目目录内: {value}")
    return candidate
