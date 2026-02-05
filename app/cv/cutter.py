from __future__ import annotations

import inspect
import logging
import os
import warnings
from typing import List, Tuple

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_onednn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_enable_new_ir", "0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_minloglevel", "2")
os.environ.setdefault("FLAGS_log_level", "3")
os.environ.setdefault("PPOCR_LOG_LEVEL", "ERROR")
os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")
os.environ.setdefault("KMP_WARNINGS", "0")

import cv2
import numpy as np

from .gpu import _ensure_cuda_bin_in_path, require_gpu, vram_limit_mb

_ensure_cuda_bin_in_path()

from paddleocr import PaddleOCR
def _configure_quiet_logs() -> None:
    for name, level in (
        ("ppocr", logging.ERROR),
        ("paddle", logging.ERROR),
        ("paddlex", logging.ERROR),
    ):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
    warnings.filterwarnings("ignore", message="No ccache found.*")


_configure_quiet_logs()

_OCR_EXT = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _use_gpu() -> bool:
    require_gpu()
    return True


def _filter_kwargs(kwargs: dict) -> dict:
    try:
        params = inspect.signature(PaddleOCR.__init__).parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        return kwargs


def _ocr_call(ocr: PaddleOCR, image):
    try:
        return ocr.ocr(image, cls=True)
    except TypeError:
        return ocr.ocr(image)


def get_ocr_ext() -> PaddleOCR:
    global _OCR_EXT
    if _OCR_EXT is not None:
        return _OCR_EXT
    require_gpu()
    use_angle = os.getenv("CV_OCR_USE_ANGLE", "0") == "1"
    det_limit = _env_int("CV_OCR_DET_LIMIT", 640)
    kwargs = _filter_kwargs(
        {
            "use_angle_cls": use_angle,
            "lang": "ru",
            "show_log": False,
            "det_db_score_mode": "fast",
            "det_db_box_thresh": 0.4,
            "use_gpu": True,
            "gpu_mem": vram_limit_mb(2000),
            "det_limit_side_len": det_limit,
            "det_limit_type": "max",
        }
    )
    _OCR_EXT = PaddleOCR(**kwargs)
    _configure_quiet_logs()
    return _OCR_EXT


def fix_leaked_letters(text: str) -> str:
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "L": "1",
        "B": "8",
        "S": "5",
        "s": "5",
        "G": "6",
        "T": "7",
        "Z": "2",
        "z": "2",
    }
    has_digits = any(char.isdigit() for char in text)
    is_short = len(text) <= 3

    if has_digits or is_short:
        res = list(text)
        for i, char in enumerate(res):
            if char in replacements:
                res[i] = replacements[char]
        return "".join(res)
    return text


def merge_labels(labels: List[dict], x_threshold: int = 50, y_threshold: int = 35) -> List[dict]:
    if not labels:
        return []

    labels.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    merged_raw = []
    while labels:
        curr = labels.pop(0)
        i = 0
        while i < len(labels):
            other = labels[i]
            c_x1, c_y1, c_x2, c_y2 = curr["bbox"]
            o_x1, o_y1, o_x2, o_y2 = other["bbox"]

            is_same_line = abs(c_y1 - o_y1) < 12
            horizontal_join = is_same_line and -10 < (o_x1 - c_x2) < x_threshold

            y_dist = o_y1 - c_y2
            overlap_width = min(c_x2, o_x2) - max(c_x1, o_x1)
            vertical_join = (overlap_width > min(c_x2 - c_x1, o_x2 - o_x1) * 0.5) and 0 <= y_dist < y_threshold

            if horizontal_join or vertical_join:
                curr["text"] = fix_leaked_letters(f"{curr['text']} {other['text']}")
                curr["bbox"] = [min(c_x1, o_x1), min(c_y1, o_y1), max(c_x2, o_x2), max(c_y2, o_y2)]
                labels.pop(i)
                i = 0
            else:
                i += 1
        merged_raw.append(curr)

    final_compact = []
    for item in merged_raw:
        x1, y1, x2, y2 = item["bbox"]
        final_compact.append(
            {
                "txt": item["text"].strip(),
                "cnt": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "wh": [int(x2 - x1), int(y2 - y1)],
            }
        )
    return final_compact


def clean_diagram_v3(
    img_input: str | np.ndarray,
    output_dir: str | None = None,
    debug: bool = False,
) -> Tuple[List[dict], np.ndarray]:
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()
    if img is None:
        return [], None 
    h, w = img.shape[:2]

    scale_factor = max(1.0, _env_float("CV_OCR_SCALE", 1.5))
    upscaled = cv2.resize(
        img,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_LANCZOS4,
    )
    result = _ocr_call(get_ocr_ext(), upscaled)

    mask = np.zeros((h, w), dtype=np.uint8)
    raw_labels = []

    if result and result[0]:
        for line in result[0]:
            text_content = fix_leaked_letters(line[1][0])
            score = line[1][1]
            if len(text_content) < 1:
                continue

            poly_points = np.array(line[0], dtype=np.float32)
            poly_orig = (poly_points / scale_factor).astype(np.int32)
            x1, y1 = np.min(poly_orig, axis=0)
            x2, y2 = np.max(poly_orig, axis=0)

            raw_labels.append(
                {
                    "text": text_content,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": score,
                }
            )
            cv2.fillPoly(mask, [poly_orig], 255)

    final_labels = merge_labels(raw_labels)

    clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    clean_img[mask > 0] = (255, 255, 255)

    if debug and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "2_text_removed.png"), clean_img)

    return final_labels, clean_img
