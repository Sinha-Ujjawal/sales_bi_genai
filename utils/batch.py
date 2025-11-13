from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def batched(items: Iterable[T], batch_size: int) -> Iterable[Sequence[T]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
