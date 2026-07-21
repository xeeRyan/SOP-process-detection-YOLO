from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import ROOT
from scripts.utils import ensure_dir


DEFAULT_IMAGES_DIR = ROOT / "datasets" / "sop" / "raw_frames"
DEFAULT_LABELS_DIR = ROOT / "datasets" / "labels"
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "sop"
DEFAULT_CLASSES = ["bearing", "cover", "tool"]


# classes.txt 是可选文件；没有时使用当前 SOP 默认类别。
def read_classes(classes_path: Path) -> list[str]:
    """读取 YOLO 类别名；没有 classes.txt 时使用项目默认类别。"""

    if not classes_path.exists():
        return DEFAULT_CLASSES
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


# 将 raw_frames 和 labels 整理为 Ultralytics YOLO 可训练目录。
def prepare_dataset(
    images_dir: str | Path = DEFAULT_IMAGES_DIR,
    labels_dir: str | Path = DEFAULT_LABELS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """整理 YOLO 训练数据集。

    输入：
    - images_dir: 已抽帧图片目录。
    - labels_dir: 人工标注后的 YOLO txt 标签目录。
    - output_dir: YOLO train/val 数据集输出目录。
    - val_ratio: 验证集比例。

    返回：
    - train/val 数量、类别列表和 sop.yaml 路径。
    """

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        raise RuntimeError(f"未找到图片: {images_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []
    for image_path in image_files:
        label_path = labels_dir / image_path.with_suffix(".txt").name
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path.name)

    if missing_labels:
        raise RuntimeError(f"存在未标注图片: {missing_labels[:10]} ... total={len(missing_labels)}")

    random.Random(seed).shuffle(pairs)
    val_count = max(1, int(round(len(pairs) * val_ratio)))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    # 输出目录符合 Ultralytics 的 images/train、labels/train 约定。
    for split in ("train", "val"):
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, label_path in split_pairs:
            shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
            shutil.copy2(label_path, output_dir / "labels" / split / label_path.name)

    classes = read_classes(labels_dir / "classes.txt")

    # sop.yaml 是 yolo train 命令直接使用的数据集配置文件。
    yaml_text = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    yaml_text.extend(f"  {idx}: {name}" for idx, name in enumerate(classes))
    (output_dir / "sop.yaml").write_text("\n".join(yaml_text) + "\n", encoding="utf-8")

    return {
        "train": len(train_pairs),
        "val": len(val_pairs),
        "classes": classes,
        "yaml": str(output_dir / "sop.yaml"),
    }


# prepare_dataset.py 的命令行参数。
def build_parser() -> argparse.ArgumentParser:
    """构建数据集整理脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="整理 YOLO 训练数据集")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES_DIR), help="原始图片目录")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_DIR), help="YOLO 标签目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="输出数据集目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser


# 命令行入口，输出训练集/验证集数量和配置文件路径。
def main() -> None:
    """命令行入口：生成 YOLO train/val 数据集和 sop.yaml。"""

    args = build_parser().parse_args()
    result = prepare_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"数据集整理完成: train={result['train']}, val={result['val']}")
    print(f"类别: {', '.join(result['classes'])}")
    print(f"配置文件: {result['yaml']}")


if __name__ == "__main__":
    main()
