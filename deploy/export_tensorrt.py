from __future__ import annotations

import argparse
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_PATH = ROOT / "models" / "best_yolo26s.onnx"
DEFAULT_ENGINE_PATH = ROOT / "models" / "best_yolo26s.engine"


def resolve_project_path(path: str | Path | None) -> Path | None:
    """Resolve relative paths from the project root or packaged _internal root."""

    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def find_trtexec(trtexec_path: str) -> str:
    """Find trtexec executable from explicit path or system PATH."""

    explicit = Path(trtexec_path)
    if explicit.exists():
        return str(explicit)

    found = shutil.which(trtexec_path)
    if found:
        return found

    raise FileNotFoundError(
        "trtexec was not found. Install NVIDIA TensorRT and add TensorRT\\bin to PATH, "
        "or pass the full trtexec.exe path with --trtexec."
    )


@lru_cache(maxsize=8)
def get_trtexec_help(trtexec_path: str) -> str:
    """Read trtexec help text once so option support can be detected."""

    completed = subprocess.run(
        [trtexec_path, "--help"],
        check=False,
        text=True,
        capture_output=True,
        errors="ignore",
    )
    return f"{completed.stdout}\n{completed.stderr}"


def trtexec_supports_option(trtexec_path: str, option: str) -> bool:
    """Return whether the installed trtexec exposes a command-line option."""

    return option in get_trtexec_help(trtexec_path)


def build_trtexec_command(
    onnx_path: Path,
    engine_path: Path,
    trtexec_path: str,
    fp16: bool = True,
    workspace_mb: int | None = None,
    verbose: bool = False,
    detect_options: bool = True,
) -> list[str]:
    """Build trtexec conversion command."""

    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    if fp16:
        if not detect_options or trtexec_supports_option(trtexec_path, "--fp16"):
            command.append("--fp16")
        else:
            print("TensorRT warning: current trtexec does not support --fp16, using default precision.")
    if workspace_mb is not None:
        command.append(f"--memPoolSize=workspace:{int(workspace_mb)}")
    if verbose:
        command.append("--verbose")
    return command

def export_tensorrt(
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    engine_path: str | Path = DEFAULT_ENGINE_PATH,
    fp16: bool = True,
    workspace_mb: int | None = None,
    trtexec_path: str = "trtexec",
    verbose: bool = False,
    dry_run: bool = False,
) -> Path:
    """Convert an ONNX model to TensorRT engine."""

    onnx_path = resolve_project_path(onnx_path) or DEFAULT_ONNX_PATH
    engine_path = resolve_project_path(engine_path) or DEFAULT_ENGINE_PATH

    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"Input model must be a .onnx file: {onnx_path}")
    if engine_path.suffix.lower() not in {".engine", ".plan"}:
        raise ValueError(f"Output path should be a .engine or .plan file: {engine_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    trtexec = trtexec_path if dry_run else find_trtexec(trtexec_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_trtexec_command(
        onnx_path=onnx_path,
        engine_path=engine_path,
        trtexec_path=trtexec,
        fp16=fp16,
        workspace_mb=workspace_mb,
        verbose=verbose,
        detect_options=not dry_run,
    )

    print("TensorRT conversion command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if dry_run:
        return engine_path

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"trtexec conversion failed, return code: {completed.returncode}")
    if not engine_path.exists():
        raise RuntimeError(f"trtexec finished, but engine file was not found: {engine_path}")
    return engine_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for TensorRT conversion."""

    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine with trtexec")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX_PATH), help="Input ONNX model path")
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE_PATH), help="Output TensorRT engine path")
    parser.add_argument("--trtexec", default="trtexec", help="trtexec.exe path or command name")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument("--workspace-mb", type=int, default=None, help="TensorRT workspace size in MB")
    parser.add_argument("--verbose", action="store_true", help="Print verbose trtexec logs")
    parser.add_argument("--dry-run", action="store_true", help="Print conversion command without running it")
    return parser


def main() -> None:
    """CLI entry: run TensorRT engine conversion."""

    args = build_parser().parse_args()
    engine_path = export_tensorrt(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=not args.fp32,
        workspace_mb=args.workspace_mb,
        trtexec_path=args.trtexec,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"dry-run completed, expected output: {engine_path}")
    else:
        print(f"TensorRT engine export completed: {engine_path}")


if __name__ == "__main__":
    main()

