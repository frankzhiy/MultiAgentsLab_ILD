from __future__ import annotations

import os


def call_chatanywhere(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    response_format: dict | None = None,
    timeout: int | None = None,
) -> str:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    api_key = os.getenv("CHATANYWHERE_API_KEY")
    base_url = os.getenv("CHATANYWHERE_BASE_URL")

    if not api_key:
        raise RuntimeError("CHATANYWHERE_API_KEY is not configured in .env")
    if not base_url:
        raise RuntimeError("CHATANYWHERE_BASE_URL is not configured in .env")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    if response_format is not None:
        request_kwargs["response_format"] = response_format

    completion = client.chat.completions.create(**request_kwargs)

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("ChatAnywhere returned an empty response")
    return content
