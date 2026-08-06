"""SOPAID Python服务入口。

提供命令行调用和“一次连接、一次请求”的TCP协议，业务命令统一交给
``task_dispatcher.run_task``，本模块不包含训练或推理业务逻辑。
"""

from __future__ import annotations

import json
import multiprocessing
import socket
import sys
import traceback
from typing import Any

from scripts.config import DEFAULT_TCP_BUFFER_SIZE
from task_dispatcher import get_tcp_config, read_json, resolve_project_path, run_task


BUFSIZE = DEFAULT_TCP_BUFFER_SIZE


def make_response(status: str, data: Any = None, message: str = "") -> bytes:
    """构建 TCP 返回消息，统一返回 UTF-8 JSON。"""

    response = {
        "status": status,
        "message": message,
        "data": data if data is not None else {},
    }
    return json.dumps(response, ensure_ascii=False).encode("utf-8")


def parse_request_bytes(request_bytes: bytes) -> dict[str, Any]:
    """解析软件端通过 TCP 传入的任务消息。"""

    decoded = request_bytes.decode("utf-8").strip()
    if decoded.startswith(("model_file:", "json_file:")):
        json_path = resolve_project_path(decoded.split(":", 1)[1])
        return read_json(json_path)
    return parse_inline_request(decoded)


def parse_inline_request(text: str) -> dict[str, Any]:
    """解析命令行或 TCP 直接传入的任务 JSON。"""

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text in {"health", "detect", "train"}:
        return {"command": text}

    # 兼容 PowerShell 将 {"command":"health"} 传成 {command:health} 的简单场景。
    if text.startswith("{") and text.endswith("}"):
        body = text[1:-1].strip()
        result: dict[str, Any] = {}
        for item in body.split(","):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            key = key.strip().strip("\\\"'")
            value = value.strip().strip("\\\"'")
            if value.lower() == "true":
                result[key] = True
            elif value.lower() == "false":
                result[key] = False
            elif value.lower() == "null":
                result[key] = None
            else:
                result[key] = value
        if result:
            return result

    raise ValueError(f"无法解析任务参数: {text}")


def handle_client(client: socket.socket, client_addr) -> str:
    """处理一个请求后主动关闭连接，与WPF客户端的一请求一连接保持一致。"""

    del client_addr  # 保留参数以兼容已有调用方。
    try:
        request_bytes = client.recv(BUFSIZE)
        if not request_bytes or request_bytes == b"end":
            return ""
        if request_bytes == b"close":
            return "close"

        dict_data = parse_request_bytes(request_bytes)
        result = run_task(dict_data)
        client.sendall(make_response("ok", result))
        return ""
    except Exception as exc:
        print(traceback.format_exc())
        try:
            client.sendall(make_response("error", message=str(exc)))
        except OSError:
            pass
        return ""
    finally:
        client.close()


def Start_tcp(cmd: str = "open_tcp", host: str | None = None, port: int | None = None) -> None:
    """启动 TCP 服务，调用方式与 DEEPLEARN_PYD.py 保持兼容。"""

    config_host, config_port = get_tcp_config()
    ip_port = (host or config_host, int(port or config_port))
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Windows 的 SO_REUSEADDR 会允许多个 SOP_PYD 实例同时监听同一端口，
    # 请求随后会被不同旧进程分流，并争用相同的输出文件。Windows 使用独占
    # 地址；其他平台保留快速重启所需的 REUSEADDR。
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(1.0)
    server.bind(ip_port)
    server.listen(1)
    print(f"SOP_PYD tcp listen: {ip_port[0]}:{ip_port[1]}")
    try:
        while True:
            try:
                client, client_addr = server.accept()
            except socket.timeout:
                continue
            print("listen a client")
            command = handle_client(client, client_addr)
            if command == "close":
                break
            print("continue")
    except KeyboardInterrupt:
        print("tcp server stopped by Ctrl+C")
    finally:
        server.close()


def get_instance(dict_data: dict[str, Any]) -> dict[str, Any]:
    """命令行直接调用算法任务。"""

    print("enter sop dll")
    result = run_task(dict_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("quit sop dll")
    return result


def main_instance() -> None:
    """正式 exe 入口，根据命令行参数选择 TCP、JSON 字符串或 JSON 文件调用方式。"""

    print("main start")
    args = sys.argv
    arg_count = len(args)
    if arg_count == 1:
        host, port = get_tcp_config()
        print(f"no args, start tcp server with default address {host}:{port}")
        Start_tcp(host=host, port=port)
        return

    if args[1] == "tcp":
        host, default_port = get_tcp_config()
        port = default_port if arg_count < 3 else int(args[2])
        Start_tcp(host=host, port=port)
    elif args[1] == "health" and arg_count == 2:
        get_instance({"command": "health"})
    elif args[1] in ("detect", "train") and arg_count == 3:
        dict_data = parse_inline_request(args[2])
        dict_data.setdefault("command", args[1])
        get_instance(dict_data)
    elif args[1] in ("detect_file", "train_file") and arg_count == 3:
        json_path = resolve_project_path(args[2])
        dict_data = read_json(json_path)
        default_command = "detect" if args[1] == "detect_file" else "train"
        dict_data.setdefault("command", default_command)
        get_instance(dict_data)
    else:
        raise ValueError(
            "unsupported startup arguments: tcp [port] / health / detect <json> / train <json> / "
            "detect_file <path> / train_file <path>"
        )
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_instance()
