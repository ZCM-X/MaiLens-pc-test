#!/usr/bin/env python3
"""Validate, train, and export a MaiLens YOLO detector.

This wrapper deliberately requires a local weights file.  Passing a model name
such as ``yolo11n.pt`` to Ultralytics can silently start a network download,
which is surprising on the offline Windows training machine.  Copy a
pretrained ``.pt`` file into the repository (or pass an absolute path) and use
``--weights``.

Examples::

    # Check the geometry labels without importing torch or starting training.
    python tools/train_detector.py --dataset datasets/maimoller-geometry/dataset.yaml \
        --weights models/yolo11n.pt --check-only

    # Train and export the best checkpoint to ONNX.
    python tools/train_detector.py --dataset datasets/maimoller-geometry/dataset.yaml \
        --weights models/yolo11n.pt --epochs 80 --imgsz 640 --device 0 \
        --project runs --name geometry --export onnx
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
}


class DatasetValidationError(ValueError):
    """Raised when the dataset cannot be safely handed to Ultralytics."""


@dataclass(frozen=True)
class DatasetSummary:
    """Counts and class statistics produced by :func:`validate_dataset`."""

    yaml_path: Path
    root: Path
    names: tuple[str, ...]
    train_images: int
    val_images: int
    train_labels: int
    val_labels: int
    missing_labels: int
    empty_labels: int
    class_counts: tuple[int, ...]

    @property
    def total_images(self) -> int:
        return self.train_images + self.val_images


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a dataset YAML with a useful dependency error."""

    try:
        import yaml  # type: ignore
    except ImportError as error:  # pragma: no cover - ultralytics normally brings this in
        raise DatasetValidationError(
            "读取 dataset.yaml 需要 PyYAML；请在当前环境安装 ultralytics 或 pyyaml。"
        ) from error
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetValidationError(f"无法读取数据集 YAML：{path}：{error}") from error
    except Exception as error:
        raise DatasetValidationError(f"dataset.yaml 格式无效：{path}：{error}") from error
    if not isinstance(value, dict):
        raise DatasetValidationError(f"dataset.yaml 顶层必须是对象：{path}")
    return value


def _normalise_names(value: Any, nc: Any) -> tuple[str, ...]:
    """Return class names while accepting Ultralytics list and dict forms."""

    if isinstance(value, dict):
        try:
            ordered = [value[key] for key in sorted(value, key=lambda item: int(item))]
        except (TypeError, ValueError, KeyError):
            ordered = [value[key] for key in sorted(value)]
        value = ordered
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        if nc is None:
            raise DatasetValidationError("dataset.yaml 必须提供非空 names（或 nc）。")
        try:
            count = int(nc)
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(f"dataset.yaml 的 nc 无效：{nc!r}") from error
        if count <= 0:
            raise DatasetValidationError(f"dataset.yaml 的 nc 必须大于 0：{count}")
        value = [str(index) for index in range(count)]
    names = tuple(str(item).strip() for item in value)
    if any(not item for item in names):
        raise DatasetValidationError("dataset.yaml 的 names 不能包含空类别名。")
    if len(set(names)) != len(names):
        raise DatasetValidationError(f"dataset.yaml 的 names 有重复类别：{names!r}")
    if nc is not None:
        try:
            declared = int(nc)
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(f"dataset.yaml 的 nc 无效：{nc!r}") from error
        if declared != len(names):
            raise DatasetValidationError(
                f"dataset.yaml 的 nc={declared} 与 names 数量={len(names)} 不一致。"
            )
    return names


def _resolve_root(yaml_path: Path, value: Any) -> Path:
    if value is None:
        return yaml_path.parent.resolve()
    root = Path(str(value)).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _expand_split(spec: Any, root: Path) -> list[Path]:
    """Expand a YAML split entry into image files, preserving deterministic order."""

    if isinstance(spec, (list, tuple)):
        result: list[Path] = []
        for item in spec:
            result.extend(_expand_split(item, root))
        return sorted(set(result))
    if spec is None:
        return []
    raw = str(spec).strip()
    if not raw:
        return []
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    # Ultralytics accepts a text file containing one image per line.
    if path.is_file() and path.suffix.lower() in {".txt", ".list"}:
        images: list[Path] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise DatasetValidationError(f"无法读取数据集图片列表：{path}：{error}") from error
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                images.extend(_expand_split(line, root))
        return sorted(set(images))
    if any(char in raw for char in "*?[]"):
        return sorted(
            Path(item).resolve()
            for item in glob.glob(str(path), recursive=True)
            if Path(item).is_file() and Path(item).suffix.lower() in IMAGE_EXTENSIONS
        )
    if path.is_dir():
        return sorted(
            item.resolve()
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path.resolve()]
    return []


def _label_candidates(image: Path, root: Path) -> Iterable[Path]:
    """Yield likely label paths for an image in common YOLO layouts."""

    yield image.with_suffix(".txt")
    parts = list(image.parts)
    # Replace the nearest ``images`` component with ``labels``.  This handles
    # both ``root/images/train/a.jpg`` and ``root/images/a.jpg``.
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            candidate = Path(*parts[:index], "labels", *parts[index + 1:]).with_suffix(".txt")
            yield candidate
            break
    try:
        relative = image.relative_to(root)
    except ValueError:
        relative = Path(image.name)
    yield (root / "labels" / relative).with_suffix(".txt")


def _find_label(image: Path, root: Path) -> Path | None:
    seen: set[Path] = set()
    for candidate in _label_candidates(image, root):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _validate_label_file(
    label_path: Path,
    class_count: int,
    class_counts: list[int],
) -> bool:
    """Validate one YOLO label and return whether it has any instances."""

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetValidationError(f"无法读取标注文件：{label_path}：{error}") from error
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        return False
    for line_number, line in enumerate(nonempty, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise DatasetValidationError(
                f"标注格式错误：{label_path}:{line_number} 应为 `class x y w h` 五列，实际 {len(fields)} 列。"
            )
        try:
            class_id = int(fields[0])
            values = [float(item) for item in fields[1:]]
        except ValueError as error:
            raise DatasetValidationError(
                f"标注格式错误：{label_path}:{line_number} 包含非数字值：{line!r}"
            ) from error
        if not 0 <= class_id < class_count:
            raise DatasetValidationError(
                f"标注类别越界：{label_path}:{line_number} 的 class={class_id}，类别范围是 0..{class_count - 1}。"
            )
        if not all(math.isfinite(item) for item in values):
            raise DatasetValidationError(f"标注包含 NaN/Inf：{label_path}:{line_number}")
        x, y, width, height = values
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise DatasetValidationError(
                f"标注坐标超出 YOLO 归一化范围：{label_path}:{line_number}：{line!r}"
            )
        class_counts[class_id] += 1
    return True


def _check_split(
    split_name: str,
    images: Sequence[Path],
    root: Path,
    class_count: int,
    class_counts: list[int],
    allow_empty: bool,
) -> tuple[int, int, int]:
    if not images:
        raise DatasetValidationError(f"{split_name} 集没有找到图片，请检查 dataset.yaml 的路径。")
    missing = 0
    empty = 0
    labels = 0
    for image in images:
        label = _find_label(image, root)
        if label is None:
            missing += 1
            continue
        labels += 1
        if not _validate_label_file(label, class_count, class_counts):
            empty += 1
    if (missing or empty) and not allow_empty:
        detail = []
        if missing:
            detail.append(f"{missing} 张图片缺少同名 .txt 标注")
        if empty:
            detail.append(f"{empty} 个标注文件为空")
        raise DatasetValidationError(
            f"{split_name} 集存在空标注：" + "，".join(detail) +
            "。如果这些确实是背景图，请加 --allow-empty-labels；否则先补齐标注。"
        )
    return labels, missing, empty


def load_dataset_spec(dataset_yaml: str | Path) -> tuple[Path, dict[str, Any], Path, tuple[str, ...]]:
    """Load and minimally normalise a dataset YAML for callers and tests."""

    yaml_path = Path(dataset_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise DatasetValidationError(f"找不到 dataset.yaml：{yaml_path}")
    spec = _read_yaml(yaml_path)
    root = _resolve_root(yaml_path, spec.get("path"))
    names = _normalise_names(spec.get("names"), spec.get("nc"))
    return yaml_path, spec, root, names


def validate_dataset(dataset_yaml: str | Path, *, allow_empty_labels: bool = False) -> DatasetSummary:
    """Validate classes, image paths, labels, and normalized coordinates."""

    yaml_path, spec, root, names = load_dataset_spec(dataset_yaml)
    if not root.exists():
        raise DatasetValidationError(f"数据集根目录不存在：{root}")
    if "train" not in spec:
        raise DatasetValidationError("dataset.yaml 缺少 train 路径。")
    if "val" not in spec:
        raise DatasetValidationError("dataset.yaml 缺少 val 路径；请显式提供验证集。")
    train_images = _expand_split(spec.get("train"), root)
    val_images = _expand_split(spec.get("val"), root)
    class_counts = [0] * len(names)
    train_labels, train_missing, train_empty = _check_split(
        "train", train_images, root, len(names), class_counts, allow_empty_labels
    )
    val_labels, val_missing, val_empty = _check_split(
        "val", val_images, root, len(names), class_counts, allow_empty_labels
    )
    missing = train_missing + val_missing
    empty = train_empty + val_empty
    if not any(class_counts):
        raise DatasetValidationError("训练集和验证集没有任何有效目标标注。")
    absent = [f"{index}:{name}" for index, (name, count) in enumerate(zip(names, class_counts)) if count == 0]
    if absent:
        raise DatasetValidationError(
            "数据集中缺少类别实例：" + ", ".join(absent) +
            "。请补充该类别标注，或修正 dataset.yaml 的 names/nc。"
        )
    return DatasetSummary(
        yaml_path=yaml_path,
        root=root,
        names=names,
        train_images=len(train_images),
        val_images=len(val_images),
        train_labels=train_labels,
        val_labels=val_labels,
        missing_labels=missing,
        empty_labels=empty,
        class_counts=tuple(class_counts),
    )


def _resolve_local_weights(weights: str | Path) -> Path:
    path = Path(weights).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到本地预训练权重：{path}\n"
            "请把 .pt 文件复制到仓库，或用 --weights 传入绝对路径。"
        )
    if path.suffix.lower() != ".pt":
        raise ValueError(f"--weights 必须是 Ultralytics .pt 文件：{path}")
    return path


def _make_runtime_yaml(summary: DatasetSummary) -> Path:
    """Write a runtime copy with an absolute dataset root for Ultralytics.

    Ultralytics versions used on Windows may resolve ``path: .`` relative to
    the process working directory instead of the YAML file.  Validation still
    accepts the portable annotation YAML; training receives this unambiguous
    copy instead.
    """
    source = summary.yaml_path.read_text(encoding="utf-8").splitlines()
    root = summary.root.as_posix()
    output: list[str] = []
    replaced = False
    for line in source:
        if line.lstrip().startswith("path:"):
            output.append(f"path: {root}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.insert(0, f"path: {root}")
    runtime = summary.yaml_path.with_name(f".{summary.yaml_path.stem}.runtime.yaml")
    runtime.write_text("\n".join(output) + "\n", encoding="utf-8")
    return runtime


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "--data", dest="dataset", required=True, type=Path, help="YOLO dataset.yaml")
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="本地 Ultralytics 预训练 .pt 权重（脚本不会自动联网下载）",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认 100")
    parser.add_argument("--imgsz", type=int, default=640, help="训练/导出输入尺寸，默认 640")
    parser.add_argument("--device", default="cpu", help="Ultralytics device，例如 cpu、0、0,1；默认 cpu")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="训练输出根目录")
    parser.add_argument("--name", default="mailens-detector", help="训练运行名称")
    parser.add_argument("--batch", type=int, default=-1, help="batch size，默认 Ultralytics 自动设置")
    parser.add_argument("--workers", type=int, default=0, help="数据加载进程数；Windows 默认 0")
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="关闭自动混合精度；离线环境中可避免 Ultralytics 下载 AMP 检查权重",
    )
    parser.add_argument("--patience", type=int, default=50, help="早停 patience，默认 50")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--exist-ok", action="store_true", help="允许复用已有 project/name 目录")
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="允许背景图缺少标注或存在空 .txt；默认把它们作为训练前错误",
    )
    parser.add_argument("--check-only", action="store_true", help="只校验数据和权重，不启动训练")
    parser.add_argument(
        "--export",
        choices=("none", "onnx"),
        default="none",
        help="训练完成后的导出格式；选择 onnx 导出 best.pt",
    )
    parser.add_argument("--export-onnx", action="store_true", help="--export onnx 的兼容写法")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset；不传则使用 Ultralytics 默认")
    parser.add_argument("--simplify", action="store_true", help="导出 ONNX 时启用 simplify")
    return parser


def _summary_text(summary: DatasetSummary) -> str:
    classes = ", ".join(
        f"{index}:{name}={count}" for index, (name, count) in enumerate(zip(summary.names, summary.class_counts))
    )
    return (
        f"数据集校验通过：train={summary.train_images} 张，val={summary.val_images} 张，"
        f"labels={summary.train_labels + summary.val_labels} 个；类别实例：{classes}"
    )


def train(args: argparse.Namespace, summary: DatasetSummary, weights: Path) -> Path:
    """Train with Ultralytics and return the best checkpoint path."""

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "当前 Python 环境没有 ultralytics；请安装与项目兼容的 ultralytics==8.4.101。"
        ) from error
    model = YOLO(str(weights), task="detect")
    runtime_yaml = _make_runtime_yaml(summary)
    train_kwargs: dict[str, Any] = {
        "data": str(runtime_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": str(args.project),
        "name": args.name,
        "workers": args.workers,
        "amp": not args.disable_amp,
        "patience": args.patience,
        "seed": args.seed,
        "exist_ok": args.exist_ok,
    }
    if args.batch >= 1:
        train_kwargs["batch"] = args.batch
    result = model.train(**train_kwargs)
    save_dir = Path(getattr(result, "save_dir", getattr(model.trainer, "save_dir", args.project / args.name)))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        # A custom trainer may only save last.pt; fail with a concrete path.
        last = save_dir / "weights" / "last.pt"
        if last.is_file():
            return last
        raise RuntimeError(f"训练结束但没有找到 best.pt 或 last.pt：{save_dir / 'weights'}")
    return best


def export_onnx(weights: Path, args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("导出 ONNX 需要 ultralytics。") from error
    model = YOLO(str(weights), task="detect")
    kwargs: dict[str, Any] = {"format": "onnx", "imgsz": args.imgsz, "device": args.device}
    if args.opset is not None:
        kwargs["opset"] = args.opset
    if args.simplify:
        kwargs["simplify"] = True
    exported = model.export(**kwargs)
    if isinstance(exported, (str, os.PathLike)):
        output = Path(exported)
    else:
        output = weights.with_suffix(".onnx")
    if not output.is_file():
        raise RuntimeError(f"Ultralytics 报告导出完成，但找不到 ONNX 文件：{output}")
    return output.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs 必须大于 0")
    if args.imgsz < 32:
        parser.error("--imgsz 必须至少为 32")
    if args.batch == 0 or args.batch < -1:
        parser.error("--batch 应为 -1（自动）或正整数")
    if args.workers < 0 or args.patience < 0:
        parser.error("--workers/--patience 不能为负数")
    if args.opset is not None and args.opset < 1:
        parser.error("--opset 必须为正整数")
    if args.export_onnx:
        args.export = "onnx"
    try:
        summary = validate_dataset(args.dataset, allow_empty_labels=args.allow_empty_labels)
        weights = _resolve_local_weights(args.weights)
        print(_summary_text(summary))
        print(f"使用本地权重：{weights}")
        if args.check_only:
            return 0
        best = train(args, summary, weights)
        print(f"训练完成：{best}")
        if args.export == "onnx":
            exported = export_onnx(best, args)
            print(f"ONNX 导出完成：{exported}")
        return 0
    except (DatasetValidationError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
