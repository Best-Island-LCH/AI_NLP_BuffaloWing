import importlib
import inspect
import os
from typing import Any, Callable, Dict, List, Optional


class PipelineAdapter:
    def __init__(self, module_path: Optional[str] = None) -> None:
        self.module_path = module_path or os.getenv("PIPELINE_MODULE", "chatbot.pipeline")
        self._pipeline: Optional[Any] = None
        self._callable: Optional[Callable[..., Any]] = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        module = importlib.import_module(self.module_path)
        if hasattr(module, "get_pipeline"):
            self._pipeline = module.get_pipeline()
        elif hasattr(module, "Pipeline"):
            self._pipeline = module.Pipeline()
        elif hasattr(module, "pipeline"):
            self._pipeline = module.pipeline
        else:
            self._pipeline = module
        self._callable = self._resolve_callable(self._pipeline)
        if self._callable is None:
            raise RuntimeError("No callable found in pipeline module.")

    def _resolve_callable(self, obj: Any) -> Optional[Callable[..., Any]]:
        for name in ("chat", "generate", "predict", "run", "__call__"):
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn
        if callable(obj):
            return obj
        return None

    def generate(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        self._load()
        assert self._callable is not None
        response = self._invoke(self._callable, message, history)
        if isinstance(response, dict) and "response" in response:
            return str(response["response"])
        if isinstance(response, (list, tuple)) and response:
            return str(response[0])
        return str(response)

    def _invoke(
        self,
        func: Callable[..., Any],
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        names = [p.name for p in params]
        kwargs: Dict[str, Any] = {}
        args: List[Any] = []

        if "message" in names:
            kwargs["message"] = message
        elif "prompt" in names:
            kwargs["prompt"] = message
        elif "query" in names:
            kwargs["query"] = message
        elif "text" in names:
            kwargs["text"] = message
        elif params:
            args.append(message)

        if history is not None:
            if "history" in names:
                kwargs["history"] = history
            elif "messages" in names:
                kwargs["messages"] = history
            elif "chat_history" in names:
                kwargs["chat_history"] = history
            elif len(params) >= 2:
                args.append(history)

        return func(*args, **kwargs)
