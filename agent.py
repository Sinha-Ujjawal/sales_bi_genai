import logging
import os
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property, reduce
from operator import __add__
from pathlib import Path
from textwrap import dedent
from typing import Any, Generator, Iterator, Sequence

import mlflow
import mlflow.langchain as mlflow_langchain
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from load_dotenv import load_dotenv
from mem0 import Memory

import db
import utils.mlflow_ollama as mlflow_ollama
import utils.patch_mem0 as _  # noqa
from app_config import AppConfig
from utils.batch import batched
from utils.date import get_last_n_quarters
from utils.mlflow_utils import mlflow_trace
from utils.sql import parse_sql

load_dotenv(override=True)

mlflow_ollama.autolog()
mlflow_langchain.autolog()

logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    model_repr: str
    tag: str
    messages: Sequence[AIMessage]


@dataclass(slots=True)
class AgentResponse:
    thread_id: str | None
    mlflow_run_id: str
    message: str
    ai_response: AIResponse | None = None
    data: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(slots=True)
class QueryAndData:
    data: dict[str, str]


@dataclass(slots=True)
class QueryAndError:
    err: dict[str, str]


@dataclass
class Workflow:
    app: AppConfig
    user_query: str
    mlflow_run_id: str
    max_attempts: int
    batch_size: int
    thread_id: str

    @cached_property
    def mem0_memory(self) -> Memory:
        config = self.app.mem0_config
        memory = Memory(config=config)
        return memory

    @mlflow_trace
    def fetch_memory_from_mem0(self) -> str:
        memories = self.mem0_memory.search(
            self.user_query,
            user_id=self.thread_id,  # TODO: use actual user_id
            run_id=self.thread_id,
        )
        memory_list = memories["results"]
        return " ".join([mem["memory"] for mem in memory_list])

    @mlflow_trace
    def update_memory_in_mem0(self, ai_complete_response: str):
        messages = [
            {"role": "user", "content": self.user_query},
            {"role": "assistant", "content": ai_complete_response},
        ]
        self.mem0_memory.add(
            messages=messages,
            user_id=self.thread_id,  # TODO: use actual user_id
            run_id=self.thread_id,
        )

    @staticmethod
    def additional_context() -> str:
        todays_date = date.today()
        date_format = "%d-%b-%Y"
        todays_date_str = todays_date.strftime(date_format)
        n = 4
        last_n_quarter_details = "; ".join(
            f"Q{quarter}, {year} ({start_date.strftime(date_format)} - {end_date.strftime(date_format)})"
            for quarter, year, start_date, end_date in get_last_n_quarters(
                todays_date, n=n
            )
        )
        additional_context = dedent(f"""
        - Database: Sqlite3
        - Today's Date: {todays_date_str}
        - Date Ranges for Last {n} quarters: {last_n_quarter_details}. Note that this is just showing the last 4 quarter date ranges for your reference.
        - FY stands for Full Year
        - YTD stands for Year To Date
        """)
        return additional_context

    def generate_sql_plan_using_llm(
        self, memory: str
    ) -> Iterator[tuple[str, AIMessage]]:
        prompt_template = PromptTemplate.from_template(
            Path("./prompts/generate_sql_plan_prompt_template.txt").read_text()
        )
        llm = ChatOllama(
            model=os.environ.get("SQL_PLAN_LLM", "qwen3:8b"),
            reasoning=os.environ.get("SQL_PLAN_LLM_REASONING", "0") == "1",
            temperature=float(os.environ.get("SQL_PLAN_LLM_TEMPERATURE", "0.0")),
            num_ctx=int(os.environ.get("SQL_PLAN_LLM_NUM_CTX", "20000")),
        )
        chain = prompt_template.pipe(llm)
        llm_repr = repr(llm)
        yield from map(
            lambda msg: (llm_repr, msg),
            chain.stream(
                {
                    "schema_ddl": self.app.db_config.schema,
                    "additional_context": self.additional_context(),
                    "table_statistics": self.app.db_config.table_stats,
                    "user_query": self.user_query,
                    "memory": memory,
                }
            ),
        )

    def generate_sql_queries_using_llm(
        self,
        *,
        sql_plan: str,
        query_and_error: QueryAndError | None = None,
    ) -> Iterator[tuple[str, AIMessage]]:
        prompt_template = PromptTemplate.from_template(
            Path("./prompts/generate_sql_queries_prompt_template.txt").read_text()
        )
        llm = ChatOllama(
            model=os.environ.get("SQL_LLM", "qwen3:8b"),
            reasoning=os.environ.get("SQL_LLM_REASONING", "0") == "1",
            temperature=float(os.environ.get("SQL_LLM_TEMPERATURE", "0.0")),
            num_ctx=int(os.environ.get("SQL_LLM_NUM_CTX", "20000")),
        )
        chain = prompt_template.pipe(llm)
        bi_notes = sql_plan
        if query_and_error is not None:
            previous_try_errors = "\n\n".join(
                f"<error>\n{err}\n</error>" for _, err in query_and_error.err.items()
            )
        else:
            previous_try_errors = ""
        llm_repr = repr(llm)
        yield from map(
            lambda msg: (llm_repr, msg),
            chain.stream(
                {
                    "schema_ddl": self.app.db_config.schema,
                    "additional_context": self.additional_context(),
                    "table_statistics": self.app.db_config.table_stats,
                    "bi_notes": bi_notes,
                    "previous_try_errors": previous_try_errors,
                    "user_query": self.user_query,
                }
            ),
        )

    def fix_sql_queries_using_llm(
        self,
        query_and_error: QueryAndError,
    ) -> Iterator[tuple[str, AIMessage]]:
        fix_sql_prompt_template = PromptTemplate.from_template(
            Path("./prompts/fix_sql_queries_prompt_template.txt").read_text()
        )
        llm = ChatOllama(
            model=os.environ.get("FIX_SQL_LLM", "qwen3:8b"),
            reasoning=os.environ.get("FIX_SQL_LLM_REASONING", "0") == "1",
            temperature=float(os.environ.get("SQL_LLM_TEMPERATURE", "0.0")),
            num_ctx=int(os.environ.get("FIX_SQL_LLM_NUM_CTX", "20000")),
        )
        fix_sql_chain = fix_sql_prompt_template.pipe(llm)
        llm_repr = repr(llm)
        sql_errors = "\n\n".join(
            f"<error>\n{err}\n</error>" for _, err in query_and_error.err.items()
        )
        yield from map(
            lambda msg: (llm_repr, msg),
            fix_sql_chain.stream(
                {
                    "schema_ddl": self.app.db_config.schema,
                    "sql_errors": sql_errors,
                    "additional_context": self.additional_context(),
                }
            ),
        )

    def summarize_user_query_and_agent_response_using_llm(
        self, agent_response: str
    ) -> Iterator[tuple[str, AIMessage]]:
        summarize_prompt_template = PromptTemplate.from_template(
            Path("./prompts/summarize_prompt_template.txt").read_text()
        )
        llm = ChatOllama(
            model=os.environ.get("SUMMARY_LLM", "qwen3:8b"),
            reasoning=os.environ.get("SUMMARY_LLM_REASONING", "0") == "1",
            temperature=float(os.environ.get("SUMMARY_LLM_TEMPERATURE", "0.0")),
            num_ctx=int(os.environ.get("SUMMARY_LLM_NUM_CTX", "20000")),
        )
        summarize_chain = summarize_prompt_template.pipe(llm)
        text = dedent(f"""
        # User's Query:
        {self.user_query}

        # Agent's Response:
        {agent_response}
        """)
        llm_repr = repr(llm)
        yield from map(
            lambda msg: (llm_repr, msg), summarize_chain.stream({"text": text})
        )

    def qa_user_query_and_agent_response_using_llm(
        self, *, memory: str, context: str
    ) -> Iterator[tuple[str, AIMessage]]:
        qa_prompt_template = PromptTemplate.from_template(
            Path("./prompts/qa_prompt_template.txt").read_text()
        )
        llm = ChatOllama(
            model=os.environ.get("QA_LLM", "qwen3:8b"),
            reasoning=os.environ.get("QA_LLM_REASONING", "0") == "1",
            temperature=float(os.environ.get("QA_LLM_TEMPERATURE", "0.0")),
            num_ctx=int(os.environ.get("QA_LLM_NUM_CTX", "20000")),
        )
        qa_chain = qa_prompt_template.pipe(llm)
        llm_repr = repr(llm)
        yield from map(
            lambda msg: (llm_repr, msg),
            qa_chain.stream(
                {
                    "user_query": self.user_query,
                    "context": context,
                    "additional_context": self.additional_context(),
                    "memory": memory,
                }
            ),
        )

    def make_res(
        self, message: str, ai_response: AIResponse | None = None, **kwargs
    ) -> AgentResponse:
        logger.info(f"thread_id: {self.thread_id}, message: {message}")
        return AgentResponse(
            thread_id=self.thread_id,
            mlflow_run_id=self.mlflow_run_id,
            message=message,
            ai_response=ai_response,
            data=kwargs,
        )

    def sql_plan_iterator(self, memory: str) -> Generator[AgentResponse, None, str]:
        yield self.make_res(message="Generating SQL Plans using LLM")
        sql_plan = ""
        for batch in batched(
            self.generate_sql_plan_using_llm(memory=memory),
            self.batch_size,
        ):
            if not batch:
                continue
            model_repr = batch[0][0]
            messages = [msg for _, msg in batch]
            yield self.make_res(
                message="SQL Plan LLM Response",
                ai_response=AIResponse(
                    model_repr=model_repr,
                    tag="sql_plan_generation_phase",
                    messages=messages,
                ),
            )
            combined_message: AIMessage = reduce(__add__, messages)
            sql_plan += str(combined_message.content)
        return sql_plan

    def generate_sql_iterator(
        self,
        attempt: int,
        sql_plan: str,
        query_and_error: QueryAndError | None = None,
    ) -> Generator[AgentResponse, None, Sequence[str]]:
        yield self.make_res(message="Generating SQL Queries using LLM")
        content = ""
        for batch in batched(
            self.generate_sql_queries_using_llm(
                sql_plan=sql_plan,
                query_and_error=query_and_error,
            ),
            self.batch_size,
        ):
            if not batch:
                continue
            model_repr = batch[0][0]
            messages = [msg for _, msg in batch]
            yield self.make_res(
                message="SQL LLM Response",
                ai_response=AIResponse(
                    model_repr=model_repr,
                    tag=f"sql_generation_phase_attempt_{attempt}",
                    messages=messages,
                ),
            )
            combined_message: AIMessage = reduce(__add__, messages)
            content += str(combined_message.content)
        return parse_sql(
            text=content + ";\n}",
            start_marker_str="{sql",
            end_marker_str="}",
        )

    def execute_sql_and_output_iterator(
        self, *, attempt: int, queries: Sequence[str]
    ) -> Generator[AgentResponse, None, tuple[QueryAndData, QueryAndError | None]]:
        # SQL queries in LLM Output
        yield self.make_res(
            message="Querying database...", attempt=attempt, queries=queries
        )
        query_and_df_or_err: dict[str, pd.DataFrame | Exception] = (
            db.query_data_multiple(self.app.db_config.sa_engine, queries)
        )
        query_and_data: dict[str, str] = {}
        query_and_error: dict[str, str] = {}
        for query, df_or_err in query_and_df_or_err.items():
            if isinstance(df_or_err, Exception):
                logger.error(df_or_err)
                query_and_error[query] = str(df_or_err)
            else:
                df = df_or_err
                query_and_data[query] = (
                    str(df.to_markdown(floatfmt=".4f")) if len(df) else ""
                )
        yield self.make_res(
            message="Got response from database",
            query_and_data={**query_and_data},
            query_and_error={**query_and_error},
        )
        return (
            QueryAndData(query_and_data),
            QueryAndError(query_and_error) if query_and_error else None,
        )

    def fix_sql_errors_iterator(
        self, *, attempt: int, query_and_error: QueryAndError
    ) -> Generator[AgentResponse, None, Sequence[str]]:
        content = ""
        for batch in batched(
            self.fix_sql_queries_using_llm(query_and_error), self.batch_size
        ):
            if not batch:
                continue
            model_repr = batch[0][0]
            messages = [msg for _, msg in batch]
            yield self.make_res(
                message="Fixing SQL LLM Response",
                ai_response=AIResponse(
                    model_repr=model_repr,
                    tag=f"fixing_sql_phase_attempt_{attempt}",
                    messages=messages,
                ),
            )
            combined_message: AIMessage = reduce(__add__, messages)
            content += str(combined_message.content)
        return parse_sql(
            text=content + ";\n}",
            start_marker_str="{sql",
            end_marker_str="}",
        )

    def question_answering_iterator(
        self, *, memory: str, query_and_data: QueryAndData
    ) -> Generator[AgentResponse, None, str]:
        data_formatted = "\n\n".join(
            dedent(f"""Query:
            {query} 

            Data:
            ```markdown
            {data_md}
            ```
            """)
            for query, data_md in query_and_data.data.items()
        )
        context = f"\n## Data Tables: \n{data_formatted}"
        yield self.make_res(message="Summarizing...")
        content = ""
        for batch in batched(
            self.qa_user_query_and_agent_response_using_llm(
                memory=memory, context=context
            ),
            self.batch_size,
        ):
            if not batch:
                continue
            model_repr = batch[0][0]
            messages = [msg for _, msg in batch]
            yield self.make_res(
                message="QA LLM Response",
                ai_response=AIResponse(
                    model_repr=model_repr,
                    tag="question_answer_phase",
                    messages=messages,
                ),
            )
            combined_message: AIMessage = reduce(__add__, messages)
            content += str(combined_message.content)
        return content

    def run_v1(self) -> Generator[AgentResponse, None, bool]:
        yield self.make_res(message="Fetching memory...")
        memory = self.fetch_memory_from_mem0()
        yield self.make_res(message="Memory fetched", memory=memory)

        sql_plan_or_followup = yield from self.sql_plan_iterator(memory=memory)
        if not sql_plan_or_followup or "[FOLLOWUP]" in sql_plan_or_followup:
            return False
        sql_plan = sql_plan_or_followup
        queries = yield from self.generate_sql_iterator(attempt=1, sql_plan=sql_plan)
        if not queries:  # No SQL queries in LLM Output
            return True

        query_and_data: QueryAndData = QueryAndData({})
        query_and_error: QueryAndError | None = None
        for attempt in range(1, self.max_attempts + 2):
            (
                new_queries_and_data,
                query_and_error,
            ) = yield from self.execute_sql_and_output_iterator(
                attempt=attempt, queries=queries
            )
            query_and_data.data.update(new_queries_and_data.data)
            if not query_and_error:  # No errors in the queries:
                break
            # if any errors then will try to fix the queries and try again
            if attempt <= self.max_attempts:
                queries = yield from self.fix_sql_errors_iterator(
                    attempt=attempt, query_and_error=query_and_error
                )
                if not queries:  # No further queries can be generated
                    break

        if query_and_error is not None:
            return False

        combined_response = yield from self.question_answering_iterator(
            memory=memory, query_and_data=query_and_data
        )
        yield self.make_res(message="Updating Memory...")
        self.update_memory_in_mem0(ai_complete_response=combined_response)
        yield self.make_res(message="Memory updated")
        return True

    # def run_v2(self) -> Generator[AgentResponse, None, bool]:
    #     sql_plan = yield from self.sql_plan_iterator()
    #     query_and_data: QueryAndData = QueryAndData({})
    #     query_and_error: QueryAndError | None = None
    #     for attempt in range(1, self.max_attempts + 1):
    #         queries = yield from self.generate_sql_iterator(
    #             attempt=attempt,
    #             sql_plan=sql_plan,
    #             query_and_error=query_and_error,
    #         )
    #         if not queries:  # No SQL queries in LLM Output
    #             break
    #         (
    #             new_queries_and_data,
    #             query_and_error,
    #         ) = yield from self.execute_sql_and_output_iterator(
    #             attempt=attempt, queries=queries
    #         )
    #         query_and_data.data.update(new_queries_and_data.data)
    #         if not query_and_error:  # No errors in the queries:
    #             break

    #     if query_and_error is not None:
    #         return False

    #     yield from self.question_answering_iterator(query_and_data=query_and_data)
    #     return True


def run_workflow(
    *, user_query: str, thread_id: str
) -> Generator[AgentResponse, None, bool]:
    load_dotenv(override=True)
    max_attempts = int(os.environ.get("MAX_ATTEMPTS", "3"))
    assert max_attempts >= 1
    batch_size = int(os.environ.get("BATCH_SIZE", "5"))
    assert batch_size >= 1
    with mlflow.start_run(run_name=f"run_{thread_id}") as run:
        ret = yield from Workflow(
            app=AppConfig.from_yaml(
                os.environ.get("APP_CONFIG_YAML", "./conf/app/default.yaml")
            ),
            user_query=user_query,
            mlflow_run_id=run.info.run_id,
            max_attempts=max_attempts,
            batch_size=batch_size,
            thread_id=thread_id,
        ).run_v1()
    return ret
