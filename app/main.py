from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def _load_dotenv() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()

from app.logging_utils import configure_logging, new_request_id, set_request_id

configure_logging()
logger = logging.getLogger("app")

from app.cv.adapter import run_cv
from app.cv.gpu import ensure_gpu_available
from app.llm.llama import get_model
from app.parsing.graph import build_graph, build_steps, format_description
from app.parsing.schema import Diagram, parse_diagram
from app.prompts.templates import build_prompt

app = FastAPI(title="Diagram Description Service")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def request_logger(request, call_next):
    request_id = new_request_id(request.headers.get("x-request-id"))
    set_request_id(request_id)
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.exception(
            "request_failed",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": 500,
                "detail": str(exc),
            },
        )
        raise
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        status_code = response.status_code if response else 500
        logger.info(
            "request",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _description_to_rows(text: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^(\d+(?:\.\d+)*)(?:[.)-])\s*(.*)$", line)
        if not match:
            continue
        num = match.group(1)
        content = match.group(2).strip()
        if content.endswith("."):
            content = content[:-1].rstrip()
        actor = None
        actor_match = re.search(r"\(исполнитель:\s*([^)]+)\)\s*$", content, flags=re.IGNORECASE)
        if actor_match:
            actor = actor_match.group(1).strip()
            content = content[: actor_match.start()].rstrip()
        lowered = content.lower()
        if lowered.startswith("условие"):
            kind = "condition"
        elif lowered.startswith("если:"):
            kind = "branch"
        else:
            kind = "step"
        rows.append({"num": num, "text": content, "actor": actor, "kind": kind})
    return rows


def _svg_to_png(data: bytes) -> bytes:
    try:
        import cairosvg
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SVG conversion requires cairosvg. Install it with: pip install cairosvg"
        ) from exc
    return cairosvg.svg2png(bytestring=data)


def _generate_description(diagram: Diagram) -> Dict[str, Any]:
    if not diagram.nodes:
        raise ValueError("Diagram has no nodes")

    graph = build_graph(diagram)
    steps = build_steps(graph)
    prompt = build_prompt(steps)

    model = get_model()
    max_tokens = int(os.getenv("LLAMA_MAX_TOKENS", "256"))
    temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.2"))
    top_p = float(os.getenv("LLAMA_TOP_P", "0.9"))

    raw_text = model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    cleaned = _postprocess_output(raw_text)
    if _looks_like_prompt_echo(cleaned) or _looks_unrelated(cleaned, steps):
        cleaned = _fallback_description(graph, steps)
    if _needs_conditions(steps) and "\u0435\u0441\u043b\u0438:" not in cleaned.lower():
        cleaned = _fallback_description(graph, steps)

    rows = _description_to_rows(cleaned)
    has_gateway = any(step.type.lower() == "gateway" for step in steps)
    if not has_gateway and any(row.get("kind") in {"condition", "branch"} for row in rows):
        cleaned = _fallback_description(graph, steps)
        rows = _description_to_rows(cleaned)
    if not rows:
        cleaned = _fallback_description(graph, steps)
        rows = _description_to_rows(cleaned)
    return {"text": cleaned, "rows": rows}


async def _generate_async(diagram: Diagram) -> Dict[str, Any]:
    return await asyncio.to_thread(_generate_description, diagram)


async def _run_cv_async(image_path: str) -> Dict[str, Any]:
    return await asyncio.to_thread(run_cv, image_path)


def _postprocess_output(text: str) -> str:
    if not text:
        return text

    cleaned = text.strip()
    result_token = "\u0420\u0415\u0417\u0423\u041b\u042c\u0422\u0410\u0422:"
    if result_token in cleaned:
        cleaned = cleaned.split(result_token)[-1].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    step_prefix = "\u0448\u0430\u0433 "
    step_lines = [line for line in lines if line.lower().startswith(step_prefix)]
    if step_lines:
        converted = []
        for line in step_lines:
            match = re.match(r"(?i)\u0448\u0430\u0433\\s*([0-9]+(?:\\.[0-9]+)*)[:.)-]*\\s*(.*)", line)
            if match:
                num = match.group(1)
                rest = match.group(2).strip()
                converted.append(f"{num}. {rest}".strip())
            else:
                converted.append(line)
        return "\n".join(converted)

    def _is_numbered(line: str) -> bool:
        if len(line) >= 2 and line[0].isdigit() and line[1] in {".", ")"}:
            return True
        if len(line) >= 3 and line[0:2].isdigit() and line[2] in {".", ")"}:
            return True
        return False

    numbered = [line for line in lines if _is_numbered(line)]
    if numbered:
        return "\n".join(numbered)
    return cleaned


def _looks_like_prompt_echo(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    markers = [
        "узлы и переходы",
        "входные данные",
        "rules:",
        "nodes and transitions",
        "graph diagram",
        "| task |",
        "| endevent |",
    ]
    if any(marker in lowered for marker in markers):
        return True
    if not (lowered.startswith("шаг ") or lowered.startswith("1.") or lowered.startswith("1)")):
        return True
    return False


def _looks_unrelated(text: str, steps) -> bool:
    lowered = text.lower()
    banned = [
        "graphviz",
        "dot file",
        "digraph",
        "node",
        "edge",
        "diagram tab",
        "graph diagram",
    ]
    if any(token in lowered for token in banned):
        return True

    vocab = _collect_vocab(steps)
    overlap = _overlap_ratio(lowered, vocab)
    return overlap < 0.08


def _collect_vocab(steps) -> set:
    vocab = set()
    for step in steps:
        for part in (step.text or "").lower().split():
            if len(part) >= 3:
                vocab.add(part.strip(".,:;\"'()[]{}"))
    return vocab


def _overlap_ratio(text: str, vocab: set) -> float:
    if not vocab:
        return 0.0
    words = [w.strip(".,:;\"'()[]{}") for w in text.split()]
    words = [w for w in words if len(w) >= 3]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in vocab)
    return hits / max(len(words), 1)


def _needs_conditions(steps) -> bool:
    for step in steps:
        if step.type.lower() != "gateway" or not step.next:
            continue
        for _, condition in step.next:
            if condition:
                return True
    return False


def _fallback_description(graph, steps) -> str:
    return format_description(graph, steps)


@app.post("/api/describe")
async def describe_endpoint(diagram: Diagram = Body(...)) -> Dict[str, Any]:
    timeout = float(os.getenv("MAX_REQUEST_SECONDS", "20"))
    try:
        llm_start = time.perf_counter()
        result = await asyncio.wait_for(_generate_async(diagram), timeout=timeout)
        llm_ms = int((time.perf_counter() - llm_start) * 1000)
        logger.info("stage_complete", extra={"stage": "llm", "duration_ms": llm_ms})
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Processing timed out") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in {".png", ".jpg", ".jpeg", ".svg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    temp_path = None
    try:
        content = await file.read()
        suffix = ext
        if ext == ".svg":
            try:
                content = _svg_to_png(content)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            suffix = ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(content)
        timeout = float(os.getenv("MAX_REQUEST_SECONDS", "20"))

        async def _process() -> Dict[str, Any]:
            cv_start = time.perf_counter()
            cv_json: Dict[str, Any] = await _run_cv_async(temp_path)
            cv_ms = int((time.perf_counter() - cv_start) * 1000)
            logger.info("stage_complete", extra={"stage": "cv", "duration_ms": cv_ms})

            diagram = parse_diagram(cv_json)
            llm_start = time.perf_counter()
            result = await _generate_async(diagram)
            llm_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.info("stage_complete", extra={"stage": "llm", "duration_ms": llm_ms})
            return result

        return await asyncio.wait_for(_process(), timeout=timeout)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Processing timed out") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.on_event("startup")
def _warmup() -> None:
    if os.getenv("GPU_ONLY", "1") == "1":
        ensure_gpu_available()
        from app.cv.cutter import get_ocr_ext
        from app.cv.test_model import get_ocr_local, get_yolo_model

        get_yolo_model()
        get_ocr_local()
        get_ocr_ext()
    if os.getenv("LLAMA_DISABLE") == "1":
        return
    get_model()
