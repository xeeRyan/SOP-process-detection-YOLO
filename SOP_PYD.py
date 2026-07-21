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
    if decoded.startswith("model_file:"):
        json_path = resolve_project_path(decoded.split(":", 1)[1])
        return read_json(json_path)
    if decoded.startswith("json_file:"):
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
    """处理单个软件端 TCP 连接。"""

    while True:
        try:
            request_bytes = client.recv(BUFSIZE)
            print("recv", request_bytes)
            if not request_bytes or request_bytes == b"end":
                print("close recv")
                client.close()
                break
            if request_bytes == b"close":
                client.close()
                return "close"
        except Exception as exc:
            print("except disconnect", exc)
            break

        try:
            dict_data = parse_request_bytes(request_bytes)
            print("enter task", dict_data.get("command") or dict_data.get("task") or dict_data.get("type"))
            result = run_task(dict_data)
            client.sendto(make_response("ok", result), client_addr)
            print("return tcp msg")
        except Exception as exc:
            print("error task")
            print(traceback.format_exc())
            client.sendto(make_response("error", message=str(exc)), client_addr)
            print("return tcp except msg")
    return ""


def Start_tcp(cmd: str = "open_tcp", host: str | None = None, port: int | None = None) -> None:
    """启动 TCP 服务，调用方式与 DEEPLEARN_PYD.py 保持兼容。"""

    config_host, config_port = get_tcp_config()
    ip_port = (host or config_host, int(port or config_port))
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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


def train_test(json_path: str) -> dict[str, Any]:
    """使用 JSON 文件进行本地测试。"""

    dict_data = read_json(json_path)
    print(json.dumps(dict_data, indent=2, ensure_ascii=False))
    return get_instance(dict_data)


def train_test2() -> None:
    """兼容 DEEPLEARN_PYD.py 的参数分发方式。"""

    args = sys.argv
    print(args)
    if len(args) == 2:
        dict_data = json.loads(args[1])
        sys.argv.remove(args[1])
        get_instance(dict_data)
    else:
        Start_tcp()


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
        if args[1] == "detect_file":
            dict_data.setdefault("command", "detect")
        if args[1] == "train_file":
            dict_data.setdefault("command", "train")
        get_instance(dict_data)
    else:
        raise ValueError(
            "unsupported startup arguments: tcp [port] / health / detect <json> / train <json> / "
            "detect_file <path> / train_file <path>"
        )


def local_test() -> None:
    """本地调试入口，命令行传入 JSON 文件路径。"""

    args = sys.argv
    if len(args) == 1:
        return
    json_path = resolve_project_path(args[1])
    train_test(str(json_path))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_instance()
