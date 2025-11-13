from datetime import date, timedelta
from typing import Iterator, NewType

Quarter = NewType("Quarter", int)
Year = NewType("Year", int)
StartDate = NewType("StartDate", date)
EndDate = NewType("EndDate", date)


def get_current_quarter(dt: date) -> Quarter:
    return Quarter((dt.month - 1) // 3 + 1)


def get_last_quarter(dt: date) -> tuple[Quarter, Year, StartDate, EndDate]:
    current_quarter = get_current_quarter(dt)
    last_quarter = current_quarter - 1
    last_quarter_year = dt.year
    if current_quarter == 1:
        last_quarter = 4
        last_quarter_year -= 1

    last_quarter_start_month = 3 * (last_quarter - 1) + 1
    last_quartner_end_month = last_quarter_start_month + 2

    last_quarter_start_date = date(
        year=last_quarter_year, month=last_quarter_start_month, day=1
    )
    if last_quartner_end_month == 12:
        last_quarter_end_date = date(year=last_quarter_year, month=12, day=31)
    else:
        last_quarter_end_date = date(
            year=last_quarter_year, month=last_quartner_end_month + 1, day=1
        ) - timedelta(days=1)

    return (
        Quarter(last_quarter),
        Year(last_quarter_year),
        StartDate(last_quarter_start_date),
        EndDate(last_quarter_end_date),
    )


def get_last_n_quarters(
    dt: date, n: int
) -> Iterator[tuple[Quarter, Year, StartDate, EndDate]]:
    if n <= 0:
        return
    for _ in range(n):
        items = get_last_quarter(dt)
        yield items
        _, _, dt, _ = items
