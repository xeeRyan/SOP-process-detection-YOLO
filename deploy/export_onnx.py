from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_yolo26s.pt"
DEFAULT_IMG_SIZE = 640
DEFAULT_OPSET = 12


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_project_path(path: str | Path | None) -> Path | None:
    """Resolve relative paths from the project root or packaged _internal root."""

    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def default_onnx_path(model_path: Path) -> Path:
    """Use the same model filename with the .onnx suffix as default output."""

    return model_path.with_suffix(".onnx")


def export_onnx(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path | None = None,
    imgsz: int = DEFAULT_IMG_SIZE,
    opset: int = DEFAULT_OPSET,
    simplify: bool = True,
    dynamic: bool = False,
    overwrite: bool = False,
) -> Path:
    """Export a YOLO .pt model to ONNX."""

    model_path = resolve_project_path(model_path) or DEFAULT_MODEL_PATH
    output_path = resolve_project_path(output_path) or default_onnx_path(model_path)

    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Input model must be a .pt file: {model_path}")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"Output path must be a .onnx file: {output_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch weight not found: {model_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"ONNX file already exists: {output_path}. Use --overwrite to replace it.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Missing ultralytics. Please install dependencies from requirements.txt.") from exc

    ensure_dir(output_path.parent)
    model = YOLO(str(model_path))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=int(imgsz),
            opset=int(opset),
            simplify=bool(simplify),
            dynamic=bool(dynamic),
        )
    )

    if exported_path.resolve() != output_path.resolve():
        if output_path.exists() and overwrite:
            output_path.unlink()
        shutil.copy2(exported_path, output_path)

    if not output_path.exists():
        raise RuntimeError(f"ONNX export finished, but output file was not found: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for ONNX export."""

    parser = argparse.ArgumentParser(description="Export YOLO .pt model to ONNX")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Input .pt weight path")
    parser.add_argument("--output", default=None, help="Output .onnx path. If omitted, use input model name.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="Export image size")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic input shape")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX graph simplification")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ONNX file")
    return parser


def main() -> None:
    """CLI entry: export ONNX and print output path."""

    args = build_parser().parse_args()
    output_path = export_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        overwrite=args.overwrite,
    )
    print(f"ONNX export completed: {output_path}")


if __name__ == "__main__":
    main()
