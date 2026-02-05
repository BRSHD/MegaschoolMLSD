from __future__ import annotations

import os
import threading
from typing import Optional


class DummyModel:
    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.2, top_p: float = 0.9) -> str:
        return "[LLM disabled]"  


class LlamaModel:
    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
        n_batch: int,
        require_gpu: bool,
    ) -> None:
        try:
            from llama_cpp import Llama, llama_supports_gpu_offload
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("llama-cpp-python is not installed") from exc

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found at: {model_path}")

        if require_gpu and not llama_supports_gpu_offload():
            raise RuntimeError("llama-cpp-python was built without GPU offload support.")

        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 400, temperature: float = 0.2, top_p: float = 0.9) -> str:
        with self._lock:
            if hasattr(self._llm, "create_chat_completion"):
                result = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return result["choices"][0]["message"]["content"].strip()

            result = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["</s>"],
            )
            return result["choices"][0]["text"].strip()


_MODEL: Optional[object] = None
_MODEL_LOCK = threading.Lock()


def get_model() -> object:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if os.getenv("LLAMA_DISABLE") == "1":
            _MODEL = DummyModel()
            return _MODEL

        model_path = os.getenv("LLAMA_MODEL_PATH", os.path.join("models", "tinyllama.gguf"))
        n_ctx = int(os.getenv("LLAMA_CTX", "4096"))
        n_threads = int(os.getenv("LLAMA_THREADS", str(os.cpu_count() or 4)))
        gpu_only = os.getenv("GPU_ONLY", "1") == "1"
        require_gpu = os.getenv("LLAMA_REQUIRE_GPU", "1") == "1" or gpu_only
        try:
            n_gpu_layers = int(os.getenv("LLAMA_GPU_LAYERS", "-1"))
        except ValueError:
            n_gpu_layers = -1
        if require_gpu and n_gpu_layers <= 0:
            n_gpu_layers = -1
        try:
            n_batch = int(os.getenv("LLAMA_BATCH", "256"))
        except ValueError:
            n_batch = 256
        try:
            _MODEL = LlamaModel(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                require_gpu=require_gpu,
            )
        except RuntimeError as exc:
            if require_gpu or os.getenv("LLAMA_STRICT", "0") == "1":
                raise
            _MODEL = DummyModel()
            print(f"[llm] Warning: {exc}. Falling back to DummyModel.")
        return _MODEL
