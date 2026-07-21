from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_yolo26s.pt"
DEFAULT_IMG_SIZE = 640


def resolve_project_path(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def default_torchscript_path(model_path: Path) -> Path:
    return model_path.with_suffix(".torchscript")


def export_torchscript(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path | None = None,
    imgsz: int = DEFAULT_IMG_SIZE,
    optimize: bool = False,
    overwrite: bool = False,
) -> Path:
    """Export an Ultralytics training checkpoint as a LibTorch-loadable model."""

    model_path = resolve_project_path(model_path) or DEFAULT_MODEL_PATH
    output_path = resolve_project_path(output_path) or default_torchscript_path(model_path)

    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Input model must be an Ultralytics .pt checkpoint: {model_path}")
    if output_path.suffix.lower() not in {".torchscript", ".pt"}:
        raise ValueError(f"Output must use .torchscript or .pt: {output_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {model_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"TorchScript file already exists: {output_path}. Use --overwrite to replace it.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Missing ultralytics. Please install dependencies from requirements.txt.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported_path = Path(
        YOLO(str(model_path)).export(
            format="torchscript",
            imgsz=int(imgsz),
            optimize=bool(optimize),
        )
    )

    if exported_path.resolve() != output_path.resolve():
        if output_path.exists() and overwrite:
            output_path.unlink()
        shutil.copy2(exported_path, output_path)

    if not output_path.exists():
        raise RuntimeError(f"TorchScript export finished, but output file was not found: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export YOLO .pt checkpoint to TorchScript")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Input Ultralytics .pt checkpoint")
    parser.add_argument("--output", default=None, help="Output .torchscript or TorchScript .pt path")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="Export image size")
    parser.add_argument("--optimize", action="store_true", help="Enable mobile-oriented TorchScript optimization")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = export_torchscript(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        optimize=args.optimize,
        overwrite=args.overwrite,
    )
    print(f"TorchScript export completed: {output_path}")


if __name__ == "__main__":
    main()
