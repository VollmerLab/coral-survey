from pathlib import Path
from datetime import datetime
import re

import cv2
import numpy as np


def load_jpg(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def laplacian_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_sharpest(paths: list[str | Path]) -> Path:
    best, best_score = None, -1.0
    for p in paths:
        try:
            img = load_jpg(p)
            score = laplacian_variance(img)
            if score > best_score:
                best, best_score = Path(p), score
        except Exception:
            continue
    return best


def exif_datetime(path: str | Path) -> datetime | None:
    try:
        import piexif
        exif = piexif.load(str(path))
        raw = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, b"")
        if raw:
            return datetime.strptime(raw.decode(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(str(path)) as im:
            exif = im._getexif() or {}
            tag_map = {v: k for k, v in TAGS.items()}
            dt_str = exif.get(tag_map.get("DateTimeOriginal", 0), "")
            if dt_str:
                return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


_DUP_RE = re.compile(r"^(.+?)([ab])(\.[^.]+)$", re.IGNORECASE)


def group_duplicates(paths: list[Path]) -> dict[str, list[Path]]:
    """Group a/b duplicate pairs by stem, return {base_stem: [path_a, path_b]}."""
    groups: dict[str, list[Path]] = {}
    singles: list[Path] = []
    for p in sorted(paths):
        m = _DUP_RE.match(p.name)
        if m:
            base = m.group(1) + p.suffix.lower()
            groups.setdefault(base, []).append(p)
        else:
            singles.append(p)
    # singles go in as their own group
    for p in singles:
        groups.setdefault(p.name, []).append(p)
    return groups


def collect_jpgs(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    pattern = "**/*.JPG" if recursive else "*.JPG"
    paths = list(folder.glob(pattern)) + list(folder.glob(pattern.lower()))
    return sorted(set(paths))
