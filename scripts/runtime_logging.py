from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.config import DEFAULT_LOG_DIR


# 运行日志模块：为训练、检测等一次完整任务生成独立日志文件。
def create_operation_logger(operation: str, log_dir: str | Path = DEFAULT_LOG_DIR) -> tuple[logging.Logger, Path]:
    """创建本次算法任务专用日志器，并返回日志对象与文件路径。"""

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"{operation}_{timestamp}.log"
    logger_name = f"sop.{operation}.{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """以 JSON 结构写入运行事件，便于软件端或人工排查。"""

    payload = {"event": event, **{key: _to_json_value(value) for key, value in fields.items()}}
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def close_operation_logger(logger: logging.Logger) -> None:
    """关闭本次任务使用的日志文件句柄。"""

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _to_json_value(value: Any) -> Any:
    """将 Path、NumPy/Torch 标量等常见对象转换为 JSON 可写类型。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _to_json_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value
