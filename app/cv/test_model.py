from __future__ import annotations

import inspect
import logging
import os
import threading
import warnings
from typing import Tuple

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
from ultralytics import YOLO

from .gpu import _ensure_cuda_bin_in_path, require_gpu, vram_limit_mb

_ensure_cuda_bin_in_path()

from paddleocr import PaddleOCR

def _configure_quiet_logs() -> None:
    for name, level in (
        ("ppocr", logging.ERROR),
        ("paddle", logging.ERROR),
        ("paddlex", logging.ERROR),
        ("ultralytics", logging.WARNING),
    ):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
    warnings.filterwarnings("ignore", message="No ccache found.*")


_configure_quiet_logs()

_MODEL = None
_MODEL_LOCK = threading.Lock()
_OCR_LOCAL = None
_OCR_LOCK = threading.Lock()

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


def _device_id() -> int:
    return require_gpu()


def _use_gpu() -> bool:
    _device_id()
    return True


def _get_weights_path() -> str:
    if os.getenv("CV_WEIGHTS_PATH"):
        return os.getenv("CV_WEIGHTS_PATH")  # type: ignore[return-value]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "models", "cv", "best.pt")


def get_yolo_model() -> YOLO:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        weights_path = _get_weights_path()
        if not os.path.exists(weights_path):
            raise RuntimeError(f"YOLO weights not found at {weights_path}")
        _MODEL = YOLO(weights_path)
        return _MODEL


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


def get_ocr_local() -> PaddleOCR:
    global _OCR_LOCAL
    if _OCR_LOCAL is not None:
        return _OCR_LOCAL
    with _OCR_LOCK:
        if _OCR_LOCAL is not None:
            return _OCR_LOCAL
        require_gpu()
        use_angle = os.getenv("CV_OCR_USE_ANGLE", "0") == "1"
        det_limit = _env_int("CV_OCR_DET_LIMIT", 640)
        kwargs = _filter_kwargs(
            {
                "use_angle_cls": use_angle,
                "lang": "ru",
                "show_log": False,
                "use_gpu": True,
                "gpu_mem": vram_limit_mb(2000),
                "det_limit_side_len": det_limit,
                "det_limit_type": "max",
            }
        )
        _OCR_LOCAL = PaddleOCR(**kwargs)
        _configure_quiet_logs()
        return _OCR_LOCAL


def simple_text_clean(text: str) -> str:
    if not text:
        return ""
    text = " ".join([w for w in text.split() if len(w) > 1 or w.isdigit()])
    return text.strip()


def _prepare_image(image_path: str) -> Tuple[cv2.Mat, cv2.Mat, cv2.Mat]:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image")
    clean_img = img.copy()
    debug_img = img.copy()
    return img, clean_img, debug_img


def predict_and_show(
    image_path: str,
    debug: bool = False,
    output_dir: str | None = None,
) -> tuple[list[dict], cv2.Mat]:
    if not os.path.exists(image_path):
        return [], None  # type: ignore[return-value]

    img, clean_img, debug_img = _prepare_image(image_path)
    h, w = img.shape[:2]

    max_img = _env_int("CV_YOLO_IMG", 1280)
    max_dim = max(h, w)
    imgsz = min(max_img, max_dim) if max_img > 0 else max_dim
    conf = _env_float("CV_YOLO_CONF", 0.5)
    model = get_yolo_model()
    device_id = _device_id()
    try:
        results = model.predict(
            source=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device_id,
            half=True,
            verbose=False,
        )
    except TypeError:
        results = model.predict(
            source=image_path,
            conf=conf,
            imgsz=imgsz,
            device=device_id,
            verbose=False,
        )

    nodes_data: list[dict] = []
    padding = 4

    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1

            x1_p, y1_p = max(0, x1 - padding), max(0, y1 - padding)
            x2_p, y2_p = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
            node_crop = img[y1_p:y2_p, x1_p:x2_p]

            node_text = ""
            if node_crop.size > 0:
                crop_res = cv2.resize(node_crop, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                ocr_res = _ocr_call(get_ocr_local(), crop_res)
                if ocr_res and ocr_res[0]:
                    node_text = " ".join([line[1][0] for line in ocr_res[0]])

            node_text = simple_text_clean(node_text)

            nodes_data.append(
                {
                    "id": f"n{i}",
                    "type": label,
                    "txt": node_text,
                    "cnt": [center_x, center_y],
                    "wh": [width, height],
                }
            )

            if debug:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(clean_img, (x1_p, y1_p), (x2_p, y2_p), (255, 255, 255), -1)

    if debug and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "0_debug.png"), debug_img)
        cv2.imwrite(os.path.join(output_dir, "1_nodes_removed.png"), clean_img)

    return nodes_data, clean_img
