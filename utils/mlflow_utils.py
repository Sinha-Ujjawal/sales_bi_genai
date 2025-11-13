from typing import Any, Callable, ParamSpec, TypeVar

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.trace_location import TraceLocationBase

P = ParamSpec("P")
R = TypeVar("R")


def mlflow_trace(
    func: Callable[P, R] | None = None,
    name: str | None = None,
    span_type: str = SpanType.UNKNOWN,
    attributes: dict[str, Any] | None = None,
    output_reducer: Callable[[list[Any]], Any] | None = None,
    trace_destination: TraceLocationBase | None = None,
) -> Callable[P, R]:
    return mlflow.trace(
        func=func,
        name=name,
        span_type=span_type,
        attributes=attributes,
        output_reducer=output_reducer,
        trace_destination=trace_destination,
    )
