from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
import sqlalchemy as sa


def query_data(sa_engine: sa.Engine, query: str) -> pd.DataFrame:
    with sa_engine.connect() as conn:
        return pd.read_sql(query, con=conn)


def query_data_multiple(
    sa_engine: sa.Engine, queries: Iterable[str]
) -> dict[str, pd.DataFrame | Exception]:
    queries_set = set(queries)
    ret: dict[str, pd.DataFrame | Exception] = {}
    with ThreadPoolExecutor() as executor:
        futs: dict[Future, str] = {}
        for query in queries_set:
            fut = executor.submit(query_data, sa_engine=sa_engine, query=query)
            futs[fut] = query
        for fut in as_completed(futs):
            query = futs[fut]
            try:
                ret[query] = fut.result()
            except Exception as e:
                ret[query] = e
    return ret
