import logging
from functools import wraps
from typing import Callable, Iterator, TypeVar

import mlflow
from mlflow.entities.span import LiveSpan, SpanEvent, SpanStatusCode
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import safe_patch

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "ollama"

_T = TypeVar("_T")


class IteratorWrapper(Iterator[_T]):
    def __init__(self, iterator: Iterator[_T]):
        self._iterator = iterator

    def __next__(self):
        return next(self._iterator)


@autologging_integration(FLAVOR_NAME)
def autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    if disable:
        return
    import ollama
    from ollama import _client
    from ollama._client import Client

    def new_fn(
        orig_fn: Callable[..., _T | Iterator[_T]],
    ) -> Callable[..., _T | Iterator[_T]]:
        @wraps(orig_fn)
        def _inner(*args, **kwargs) -> _T | Iterator[_T]:
            ret = orig_fn(*args, **kwargs)
            if isinstance(ret, Iterator):
                return IteratorWrapper(ret)
            else:
                return ret

        return _inner

    destinations = [Client, _client, ollama]
    methods = [
        "generate",
        "chat",
        "embed",
        "embeddings",
        "pull",
        "push",
        "create",
        "delete",
        "list",
        "copy",
        "show",
        "ps",
        "web_search",
        "web_fetch",
    ]
    for dest in destinations:
        for method in methods:
            setattr(dest, method, new_fn(getattr(dest, method)))  # type: ignore
            safe_patch(FLAVOR_NAME, dest, method, patch_fn(method))


def patch_fn(fn_name: str) -> Callable[..., None]:
    def _inner(original_function, *args, **kwargs) -> None:
        _logger.debug(f"{fn_name} is called with {args=}, {kwargs=}")

        config = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else None

        _logger.debug(f"{run_id=}")

        if not config.log_traces:
            original_function(*args, **kwargs)
            return

        span = mlflow.start_span_no_context(name=fn_name)
        if run_id is not None:
            tm = InMemoryTraceManager().get_instance()
            tm.set_trace_metadata(span.trace_id, TraceMetadataKey.SOURCE_RUN, run_id)

        span.set_inputs(inputs={"args": args, "**kwargs": kwargs})
        try:
            result = original_function(*args, **kwargs)
            _logger.debug(
                f"{hasattr(result, '__iter__')=}, {isinstance(result, Iterator)=}"
            )
            if not hasattr(result, "__iter__") or not isinstance(result, Iterator):
                span.end(outputs={"result": result})
                return
            result: IteratorWrapper
            result._iterator = _end_span_with_iterable_results(  # type: ignore
                span,
                results=result._iterator,  # type: ignore
            )
        except Exception as e:
            _end_span_on_exception(span, e)
            raise

    return _inner


def _end_span_on_exception(span: LiveSpan, e: Exception) -> None:
    try:
        span.add_event(SpanEvent.from_exception(e))
        span.end(status=SpanStatusCode.ERROR)
    except Exception as inner_e:
        _logger.warning(f"Encountered unexpected error when ending trace: {inner_e}")


def _end_span_with_iterable_results(
    span: LiveSpan, results: Iterator[_T]
) -> Iterator[_T]:
    try:
        chunks = []
        for chunk in results:
            chunks.append(chunk)
            yield chunk
        span.end(outputs={"chunks": chunks}, status=SpanStatusCode.OK)
    except Exception as e:
        _end_span_on_exception(span, e)
        raise
