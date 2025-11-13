from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache

import requests
from html_to_markdown import ConversionOptions, convert


def parse_sql(
    *,
    text: str,
    start_marker_str: str = "```sql",
    end_marker_str: str = "```",
) -> list[str]:
    """
    Parses SQL queries from the provided text and returns them as a list.
    NOTE: This possibly does not work for all cases. DO NOT USE FOR GENERAL CASE
    Args:
    text: A string containing SQL queries enclosed in ```sql ``` code blocks,
          with each query ending with a semicolon.
    Returns:
    A list of SQL query strings in order of appearance.
    """
    # Find all SQL code blocks
    sql_blocks = []
    start_idx = 0
    while True:
        start_marker = text.find(start_marker_str, start_idx)
        if start_marker == -1:
            break
        end_marker = text.find(end_marker_str, start_marker + len(start_marker_str) - 1)
        if end_marker == -1:
            break
        sql_block = text[start_marker + len(start_marker_str) : end_marker].strip()
        sql_blocks.append(sql_block)
        start_idx = end_marker + len(end_marker_str)

    # Extract individual queries from each SQL block
    queries = []
    for block in sql_blocks:
        # Split by semicolon and filter out empty strings
        block_queries = [q.strip() for q in block.split(";") if q.strip()]
        queries.extend([q + ";" for q in block_queries])

    return queries


@cache
def sqlite3_docs() -> str:
    """Returns SQLITE3 Docs as markdown"""

    def convert_sqlite_doc_to_md(url: str) -> str:
        html = requests.get(url, verify=False).content.decode("utf-8")
        opts = ConversionOptions(strip_tags={"img", "svg"})
        return convert(html=html, options=opts)

    sqlite3_html_utls = [
        "https://sqlite.org/lang_corefunc.html",
        "https://sqlite.org/lang_datefunc.html",
        "https://sqlite.org/lang_aggfunc.html",
        "https://sqlite.org/windowfunctions.html",
        "https://sqlite.org/lang_mathfunc.html",
        "https://sqlite.org/json1.html",
    ]
    ret = ""
    with ThreadPoolExecutor() as executor:
        futs = [
            executor.submit(convert_sqlite_doc_to_md, url=url)
            for url in sqlite3_html_utls
        ]
        for fut in as_completed(futs):
            ret += fut.result()
    return ret
