"""TEMPORARY minimal LangChain model layer, owned by Person C in the real build.

Provides just enough (get_chat_model, build_json_chain) for Person D's
insights.py to call a real LLM. Delete/replace once Person C's full
provider-fallback chain (HF + Groq + Gemini with .with_fallbacks) lands.
"""

import json
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

from app.config import settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_chat_model():
    """Minimal single-provider model: Groq only, for now."""
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set")
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=settings.max_new_tokens,
        timeout=90,
        max_retries=2,
    )


def describe_models():
    try:
        get_chat_model()
    except LLMError as exc:
        return {"error": str(exc), "configured": settings.provider_order}
    return {
        "provider_order": settings.provider_order,
        "active": ["groq"],
        "status": "configured (temporary groq-only chain)",
    }


def parse_message(message, parser: PydanticOutputParser) -> BaseModel:
    text = message.content if isinstance(message, BaseMessage) else str(message)
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    try:
        return parser.parse(text)
    except Exception:  # noqa: BLE001
        pass
    blob = first_json_object(text)
    if blob is None:
        raise LLMError(f"model did not return JSON: {text[:200]!r}")
    try:
        return parser.pydantic_object.model_validate(blob)
    except ValidationError as exc:
        raise LLMError(f"JSON did not match {parser.pydantic_object.__name__}: {exc}") from exc


def first_json_object(text: str) -> dict | None:
    cleaned = text.strip()
    if "```" in cleaned:
        for part in cleaned.split("```"):
            candidate = part.lstrip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
            if "{" in candidate:
                cleaned = candidate
                break
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def build_json_chain(system_prompt: str, schema: type[T]) -> Runnable:
    """prompt | model | parser, returning an instance of `schema`."""
    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}\n\nRESPONSE FORMAT\n{format_instructions}"),
            ("human", "{input}"),
        ]
    ).partial(
        system_prompt=system_prompt,
        format_instructions=parser.get_format_instructions(),
    )
    return prompt | get_chat_model() | RunnableLambda(lambda msg: parse_message(msg, parser))