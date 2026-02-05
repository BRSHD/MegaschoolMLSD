from __future__ import annotations

import os
from typing import Any, Dict

from .cutter import clean_diagram_v3
from .slip_arrows import detect_orthogonal_arrows
from .test_model import predict_and_show


def _debug_enabled() -> bool:
    return os.getenv("CV_DEBUG", "0") == "1"


def _result_dir() -> str | None:
    if not _debug_enabled():
        return None
    return os.getenv("CV_RESULT_DIR")


def run_cv(source_image_path: str) -> Dict[str, Any]:
    debug = _debug_enabled()
    output_dir = _result_dir()

    final_data: Dict[str, Any] = {
        "source_file": os.path.basename(source_image_path),
        "nodes": [],
        "labels": [],
        "arrows": [],
    }

    nodes, img_nodes_removed = predict_and_show(source_image_path, debug=debug, output_dir=output_dir)
    if img_nodes_removed is None:
        raise RuntimeError("Node detection failed")

    labels, img_fully_cleaned = clean_diagram_v3(img_nodes_removed, output_dir=output_dir, debug=debug)
    if img_fully_cleaned is None:
        raise RuntimeError("Text cleanup failed")

    arrows = detect_orthogonal_arrows(img_fully_cleaned, output_dir=output_dir, debug=debug)

    final_data["nodes"] = nodes
    final_data["labels"] = labels
    final_data["arrows"] = arrows

    return final_data
