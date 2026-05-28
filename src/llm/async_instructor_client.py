"""异步版 instructor 客户端，用于并发调用 LLM 做单句分类等。

`call_with_instructor_async` 与同步版 `call_with_instructor` 接口对齐，
区别只是返回一个 awaitable。
"""

from __future__ import annotations

import os
from typing import Type, TypeVar

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


async def call_with_instructor_async(
    response_model: Type[T],
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    timeout: int | None = None,
    max_retries: int = 3,
) -> T:
    load_dotenv()

    api_key = os.getenv("CHATANYWHERE_API_KEY")
    base_url = os.getenv("CHATANYWHERE_BASE_URL")

    if not api_key:
        raise RuntimeError("CHATANYWHERE_API_KEY is not configured in .env")
    if not base_url:
        raise RuntimeError("CHATANYWHERE_BASE_URL is not configured in .env")

    raw_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    client = instructor.from_openai(raw_client)

    return await client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
    )
