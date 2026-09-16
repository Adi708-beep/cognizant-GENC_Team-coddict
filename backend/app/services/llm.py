"""
LangChain model layer.

One place decides which model Feedy talks to. Everything else asks for a chain.
Provider order comes from LLM_PROVIDER_ORDER. The first provider that can be
constructed becomes primary; the rest are attached with Runnable.with_fallbacks,
so a 429 or a dead key moves to the next provider inside a single .invoke()
call instead of raising.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

from app.config import settings


T = TypeVar("T", bound=BaseModel)

_ACTIVE: list[str] = []


VISION_PROMPT = (
    "Transcribe every piece of text on this feedback form exactly as written, "
    "including handwriting. Preserve the question labels and the answers. "
    "Do not summarise, do not correct spelling, do not add commentary. "
    "If a word is illegible, write [illegible]."
)


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------- providers


def _huggingface() -> BaseChatModel:
    """Hugging Face, either a local pipeline or a served endpoint."""

    from langchain_huggingface import ChatHuggingFace

    if settings.use_local_llm:
        from langchain_huggingface import HuggingFacePipeline

        pipe = HuggingFacePipeline.from_model_id(
            model_id=settings.hf_local_model,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": settings.max_new_tokens,
                "do_sample": False,
                "return_full_text": False,
            },
        )

        return ChatHuggingFace(
            llm=pipe,
            model_id=settings.hf_local_model,
        )

    if not settings.hf_token:
        raise LLMError("HF_TOKEN is not set")

    from langchain_huggingface import HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=settings.hf_repo_id,
        task="text-generation",
        huggingfacehub_api_token=settings.hf_token,
        max_new_tokens=settings.max_new_tokens,
        do_sample=False,
        temperature=0.01,
        repetition_penalty=1.03,
        timeout=120,
    )

    return ChatHuggingFace(
        llm=endpoint,
        model_id=settings.hf_repo_id,
    )


def _groq() -> BaseChatModel:
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


def _gemini() -> BaseChatModel:
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_output_tokens=settings.max_new_tokens,
    )


_BUILDERS = {
    "huggingface": _huggingface,
    "groq": _groq,
    "gemini": _gemini,
}


@lru_cache(maxsize=1)
def get_chat_model() -> Runnable:
    """Primary model with the remaining providers attached as fallbacks."""

    built: list[tuple[str, BaseChatModel]] = []

    for name in settings.provider_order:
        builder = _BUILDERS.get(name)

        if builder is None:
            print(
                f"[llm] unknown provider "
                f"'{name}' in LLM_PROVIDER_ORDER, skipped"
            )
            continue

        try:
            built.append((name, builder()))
        except Exception as exc:
            print(f"[llm] provider '{name}' unavailable: {exc}")

    if not built:
        raise LLMError(
            "no chat model could be built - check the keys in .env"
        )

    _ACTIVE.clear()
    _ACTIVE.extend(name for name, _ in built)

    print("[llm] model chain:", " -> ".join(_ACTIVE))

    primary = built[0][1]

    if len(built) == 1:
        return primary

    return primary.with_fallbacks(
        [model for _, model in built[1:]]
    )


def describe_models() -> dict[str, Any]:
    try:
        get_chat_model()
    except LLMError as exc:
        return {
            "error": str(exc),
            "configured": settings.provider_order,
        }

    return {
        "configured": settings.provider_order,
        "active": list(_ACTIVE),
        "primary": _ACTIVE[0] if _ACTIVE else None,
        "hf_mode": (
            "local-pipeline"
            if settings.use_local_llm
            else "inference-endpoint"
        ),
        "hf_model": (
            settings.hf_local_model
            if settings.use_local_llm
            else settings.hf_repo_id
        ),
        "groq_model": settings.groq_model,
        "gemini_model": settings.gemini_model,
    }


# ---------------------------------------------------------------- JSON chains


def build_json_chain(
    system_prompt: str,
    schema: type[T],
) -> Runnable:
    """
    prompt | model | parser, returning an instance of `schema`.

    Call it with:
        chain.invoke({"input": "..."})
    """

    parser = PydanticOutputParser(
        pydantic_object=schema
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{system_prompt}\n\n"
                "RESPONSE FORMAT\n"
                "{format_instructions}",
            ),
            (
                "human",
                "{input}",
            ),
        ]
    ).partial(
        system_prompt=system_prompt,
        format_instructions=parser.get_format_instructions(),
    )

    return (
        prompt
        | get_chat_model()
        | RunnableLambda(
            lambda msg: parse_message(msg, parser)
        )
    )


def parse_message(
    message: Any,
    parser: PydanticOutputParser,
) -> BaseModel:
    """
    Turn a model reply into a validated object, tolerating fences and prose.
    """

    text = (
        message.content
        if isinstance(message, BaseMessage)
        else str(message)
    )

    if isinstance(text, list):
        # Some providers return a list of content blocks rather than a string.
        text = "".join(
            part.get("text", "")
            for part in text
            if isinstance(part, dict)
        )

    try:
        return parser.parse(text)
    except Exception:
        # Fall through to tolerant JSON extraction.
        pass

    blob = first_json_object(text)

    if blob is None:
        raise LLMError(
            f"model did not return JSON: {text[:200]!r}"
        )

    try:
        return parser.pydantic_object.model_validate(blob)

    except ValidationError as exc:
        raise LLMError(
            f"JSON did not match "
            f"{parser.pydantic_object.__name__}: {exc}"
        ) from exc


def first_json_object(text: str) -> dict | None:
    """
    Pull the first complete JSON object out of a reply.

    Brace-counting means a trailing sentence after the closing brace
    does not break parsing.
    """

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
                    return json.loads(
                        cleaned[start:index + 1]
                    )
                except json.JSONDecodeError:
                    return None

    return None


# ---------------------------------------------------------------- vision tier


def vision_read(
    image_bytes: bytes,
    mime_type: str = "image/png",
) -> str:
    """Tier-3 OCR: hand the page image to a vision-capable chat model."""

    if not settings.gemini_api_key:
        raise LLMError(
            "vision OCR requested but GEMINI_API_KEY is not set"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_output_tokens=2048,
    )

    encoded = base64.b64encode(image_bytes).decode()

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": VISION_PROMPT,
            },
            {
                "type": "image_url",
                "image_url": (
                    f"data:{mime_type};base64,{encoded}"
                ),
            },
        ]
    )

    try:
        return str(
            model.invoke([message]).content
        ).strip()

    except Exception as exc:
        raise LLMError(
            f"vision model failed: {exc}"
        ) from exc