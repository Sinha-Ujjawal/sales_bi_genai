import json
import logging
import uuid
from enum import StrEnum
from functools import reduce
from operator import __add__
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel

import agent

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s - %(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()


def stream_objs(objs: Iterator[Any]) -> Iterator[str]:
    for obj in objs:
        yield json.dumps(obj)
        yield "\n"


class QueryRequest(BaseModel):
    user_query: str
    thread_id: str | None = None


class QueryResponseType(StrEnum):
    REASONING = "reasoning"
    CONTENT = "content"


class QueryResponse(BaseModel):
    thread_id: str
    mlflow_run_id: str
    model_repr: str
    tag: str
    response_type: QueryResponseType
    response: str


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    thread_id = request.thread_id
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    user_query = request.user_query

    def stream_generator() -> Iterator[QueryResponse]:
        for agent_response in agent.run_workflow(
            thread_id=thread_id, user_query=user_query
        ):
            ai_response = agent_response.ai_response
            if ai_response is not None:
                messages = ai_response.messages
                if not messages:
                    continue
                message: AIMessage = reduce(__add__, messages)
                additional_data = message.additional_kwargs
                if "reasoning_content" in additional_data:
                    yield QueryResponse(
                        thread_id=thread_id,
                        mlflow_run_id=agent_response.mlflow_run_id,
                        model_repr=ai_response.model_repr,
                        tag=ai_response.tag,
                        response_type=QueryResponseType.REASONING,
                        response=additional_data["reasoning_content"],
                    )
                if message.content:
                    yield QueryResponse(
                        thread_id=thread_id,
                        mlflow_run_id=agent_response.mlflow_run_id,
                        model_repr=ai_response.model_repr,
                        tag=ai_response.tag,
                        response_type=QueryResponseType.CONTENT,
                        response=str(message.content),
                    )

    stream = stream_objs(map(lambda x: x.model_dump(), stream_generator()))
    return StreamingResponse(stream, media_type="application/json")
