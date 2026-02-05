from __future__ import annotations

import importlib
import os
from typing import Any, Dict


def run_cv(image_path: str) -> Dict[str, Any]:
    module_path = os.getenv("CV_MODULE") or "app.cv.pipeline"
    func_name = os.getenv("CV_FUNCTION") or "run_cv"

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        if module_path != "app.cv.pipeline":
            try:
                module = importlib.import_module("app.cv.pipeline")
                print(
                    f"[cv] Warning: failed to import '{module_path}', "
                    "falling back to 'app.cv.pipeline'."
                )
            except Exception:
                module = None
        else:
            module = None

        if module is None:
            raise RuntimeError(
                "CV module not found. Place teammate CV code in app/cv/pipeline.py "
                "with run_cv(image_path)->dict, or set CV_MODULE/CV_FUNCTION."
            ) from exc

    try:
        func = getattr(module, func_name)
    except AttributeError as exc:
        raise RuntimeError(f"CV function '{func_name}' not found in {module_path}") from exc

    result = func(image_path)
    if not isinstance(result, dict):
        raise RuntimeError("CV function must return a dict JSON")
    return result
