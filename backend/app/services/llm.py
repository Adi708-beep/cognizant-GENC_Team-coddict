"""
Feedy LLM Service

Responsibilities:
    1. Build the configured LangChain chat model
    2. Support local TinyLlama through Hugging Face
    3. Support Hugging Face remote inference
    4. Support Groq fallback
    5. Support Gemini fallback
    6. Build Pydantic JSON chains
    7. Parse tolerant JSON model responses
    8. Provide vision OCR for extraction.py
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


# ============================================================
# TYPES
# ============================================================

T = TypeVar("T", bound=BaseModel)


# ============================================================
# STATE
# ============================================================

_ACTIVE: list[str] = []


# ============================================================
# ERRORS
# ============================================================

class LLMError(RuntimeError):
    pass


# ============================================================
# VISION PROMPT
# ============================================================

VISION_PROMPT = (
    "Transcribe every piece of text on this feedback form exactly "
    "as written, including handwriting. Preserve the question labels "
    "and the answers. Do not summarise, do not correct spelling, "
    "do not add commentary. If a word is illegible, write [illegible]."
)


# ============================================================
# HUGGING FACE
# ============================================================

def _huggingface() -> BaseChatModel:
    """
    Build the Hugging Face model.

    If USE_LOCAL_LLM=true:
        Runs TinyLlama locally.

    If USE_LOCAL_LLM=false:
        Uses Hugging Face Inference Endpoint.
    """

    from langchain_huggingface import ChatHuggingFace

    # --------------------------------------------------------
    # LOCAL HUGGING FACE MODEL
    # --------------------------------------------------------

    if settings.use_local_llm:

        from langchain_huggingface import HuggingFacePipeline

        print(
            "[llm] building local Hugging Face model:",
            settings.hf_local_model,
        )

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

    # --------------------------------------------------------
    # REMOTE HUGGING FACE
    # --------------------------------------------------------

    if not settings.hf_token:

        raise LLMError(
            "HF_TOKEN is not set"
        )

    from langchain_huggingface import HuggingFaceEndpoint

    print(
        "[llm] building Hugging Face endpoint:",
        settings.hf_repo_id,
    )

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


# ============================================================
# GROQ
# ============================================================

def _groq() -> BaseChatModel:
    """
    Build Groq model.
    """

    if not settings.groq_api_key:

        raise LLMError(
            "GROQ_API_KEY is not set"
        )

    from langchain_groq import ChatGroq

    print(
        "[llm] building Groq:",
        settings.groq_model,
    )

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=settings.max_new_tokens,
        timeout=90,
        max_retries=2,
    )


# ============================================================
# GEMINI
# ============================================================

def _gemini() -> BaseChatModel:
    """
    Build Gemini model.
    """

    if not settings.gemini_api_key:

        raise LLMError(
            "GEMINI_API_KEY is not set"
        )

    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
    )

    print(
        "[llm] building Gemini:",
        settings.gemini_model,
    )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_output_tokens=settings.max_new_tokens,
    )


# ============================================================
# PROVIDER REGISTRY
# ============================================================

_BUILDERS = {
    "huggingface": _huggingface,
    "groq": _groq,
    "gemini": _gemini,
}


# ============================================================
# CHAT MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_chat_model() -> Runnable:
    """
    Build the primary model and attach the remaining providers
    as LangChain fallbacks.

    Example:

        Hugging Face
             ↓
           fails
             ↓
           Groq
             ↓
           fails
             ↓
          Gemini
    """

    built: list[
        tuple[str, BaseChatModel]
    ] = []

    print()
    print("[llm] ========================================")
    print("[llm] BUILDING MODEL CHAIN")
    print("[llm] ========================================")

    # --------------------------------------------------------
    # Try providers in configured order
    # --------------------------------------------------------

    for name in settings.provider_order:

        builder = _BUILDERS.get(name)

        if builder is None:

            print(
                f"[llm] unknown provider '{name}', skipped"
            )

            continue

        try:

            model = builder()

            built.append(
                (name, model)
            )

            print(
                f"[llm] provider '{name}' ready"
            )

        except Exception as exc:

            print(
                f"[llm] provider '{name}' unavailable:",
                exc,
            )

    # --------------------------------------------------------
    # Nothing available
    # --------------------------------------------------------

    if not built:

        raise LLMError(
            "no chat model could be built - "
            "check the keys in .env"
        )

    # --------------------------------------------------------
    # Save active provider names
    # --------------------------------------------------------

    _ACTIVE.clear()

    _ACTIVE.extend(
        name
        for name, _ in built
    )

    print(
        "[llm] model chain:",
        " -> ".join(_ACTIVE),
    )

    # --------------------------------------------------------
    # Primary
    # --------------------------------------------------------

    primary = built[0][1]

    # --------------------------------------------------------
    # No fallback needed
    # --------------------------------------------------------

    if len(built) == 1:

        return primary

    # --------------------------------------------------------
    # Attach LangChain fallbacks
    # --------------------------------------------------------

    fallback_models = [
        model
        for _, model in built[1:]
    ]

    return primary.with_fallbacks(
        fallback_models
    )


# ============================================================
# MODEL DESCRIPTION
# ============================================================

def describe_models() -> dict[str, Any]:
    """
    Return information used by /api/health.
    """

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
        "primary": (
            _ACTIVE[0]
            if _ACTIVE
            else None
        ),
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


# ============================================================
# JSON CHAIN
# ============================================================

def build_json_chain(
    system_prompt: str,
    schema: type[T],
) -> Runnable:
    """
    Build:

        Prompt
          ↓
        Chat Model
          ↓
        JSON/Pydantic parser

    Returns a Pydantic object.

    Usage:

        chain = build_json_chain(
            SYSTEM_PROMPT,
            MyModel
        )

        result = chain.invoke({
            "input": "..."
        })
    """

    # --------------------------------------------------------
    # Pydantic parser
    # --------------------------------------------------------

    parser = PydanticOutputParser(
        pydantic_object=schema
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # parser.get_format_instructions() contains JSON braces.
    #
    # Do NOT directly concatenate them into the prompt
    # template.
    #
    # Using partial() prevents ChatPromptTemplate from
    # interpreting those braces as variables.
    # --------------------------------------------------------

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
        format_instructions=(
            parser.get_format_instructions()
        ),
    )

    # --------------------------------------------------------
    # LCEL chain
    # --------------------------------------------------------

    return (
        prompt
        | get_chat_model()
        | RunnableLambda(
            lambda message:
            parse_message(
                message,
                parser,
            )
        )
    )


# ============================================================
# PARSE MODEL RESPONSE
# ============================================================

def parse_message(
    message: Any,
    parser: PydanticOutputParser,
) -> BaseModel:
    """
    Parse model response into the requested Pydantic model.

    Handles:

        plain JSON

        ```json
        {...}
        ```

        explanatory text before JSON
    """

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    if isinstance(
        message,
        BaseMessage,
    ):

        text = message.content

    else:

        text = str(message)

    # --------------------------------------------------------
    # Some providers return content blocks
    # --------------------------------------------------------

    if isinstance(
        text,
        list,
    ):

        text = "".join(
            part.get("text", "")
            for part in text
            if isinstance(part, dict)
        )

    text = str(text).strip()

    # --------------------------------------------------------
    # First attempt: normal Pydantic parser
    # --------------------------------------------------------

    try:

        return parser.parse(
            text
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Second attempt: extract JSON object
    # --------------------------------------------------------

    blob = first_json_object(
        text
    )

    if blob is None:

        raise LLMError(
            "model did not return JSON: "
            f"{text[:300]!r}"
        )

    # --------------------------------------------------------
    # Validate extracted JSON
    # --------------------------------------------------------

    try:

        return parser.pydantic_object.model_validate(
            blob
        )

    except ValidationError as exc:

        raise LLMError(
            "JSON did not match "
            f"{parser.pydantic_object.__name__}: "
            f"{exc}"
        ) from exc


# ============================================================
# EXTRACT FIRST JSON OBJECT
# ============================================================

def first_json_object(
    text: str,
) -> dict | None:
    """
    Find the first complete JSON object.

    Uses brace counting instead of a simple regex so nested
    JSON objects work correctly.
    """

    cleaned = text.strip()

    # --------------------------------------------------------
    # Handle Markdown fences
    # --------------------------------------------------------

    if "```" in cleaned:

        for part in cleaned.split("```"):

            candidate = part.lstrip()

            if candidate.lower().startswith(
                "json"
            ):

                candidate = candidate[4:]

            if "{" in candidate:

                cleaned = candidate

                break

    # --------------------------------------------------------
    # Find first opening brace
    # --------------------------------------------------------

    start = cleaned.find("{")

    if start == -1:

        return None

    # --------------------------------------------------------
    # Brace parser
    # --------------------------------------------------------

    depth = 0

    in_string = False

    escaped = False

    for index in range(
        start,
        len(cleaned),
    ):

        char = cleaned[index]

        # ----------------------------------------------------
        # Escaped character
        # ----------------------------------------------------

        if escaped:

            escaped = False

            continue

        if char == "\\":

            escaped = True

            continue

        # ----------------------------------------------------
        # String boundaries
        # ----------------------------------------------------

        if char == '"':

            in_string = not in_string

            continue

        if in_string:

            continue

        # ----------------------------------------------------
        # Braces
        # ----------------------------------------------------

        if char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                candidate = cleaned[
                    start:index + 1
                ]

                try:

                    return json.loads(
                        candidate
                    )

                except json.JSONDecodeError:

                    return None

    return None


# ============================================================
# VISION OCR
# ============================================================

def vision_read(
    image_bytes: bytes,
    mime_type: str = "image/png",
) -> str:
    """
    Tier-3 OCR using a vision-capable model.

    This function matches extraction.py's expected interface:

        vision_read(image_bytes, mime_type)

    It intentionally uses Gemini because the local TinyLlama
    text model is NOT a vision model.
    """

    # --------------------------------------------------------
    # Gemini required
    # --------------------------------------------------------

    if not settings.gemini_api_key:

        raise LLMError(
            "vision OCR requested but "
            "GEMINI_API_KEY is not set"
        )

    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
    )

    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_output_tokens=2048,
    )

    # --------------------------------------------------------
    # Encode image
    # --------------------------------------------------------

    encoded = base64.b64encode(
        image_bytes
    ).decode()

    # --------------------------------------------------------
    # Multimodal message
    # --------------------------------------------------------

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": VISION_PROMPT,
            },
            {
                "type": "image_url",
                "image_url": (
                    f"data:{mime_type};base64,"
                    f"{encoded}"
                ),
            },
        ]
    )

    # --------------------------------------------------------
    # Call vision model
    # --------------------------------------------------------

    try:

        response = model.invoke(
            [message]
        )

        return str(
            response.content
        ).strip()

    except Exception as exc:

        raise LLMError(
            f"vision model failed: {exc}"
        ) from exc


# ============================================================
# SIMPLE GENERATE FUNCTION
# ============================================================

def generate(
    prompt: str,
    max_new_tokens: int | None = None,
) -> str:
    """
    Simple text generation helper.

    Kept for your test_local_llm.py and any other code
    that directly wants a string.
    """

    if max_new_tokens is None:

        max_new_tokens = (
            settings.max_new_tokens
        )

    print(
        "[llm] USE_LOCAL_LLM =",
        settings.use_local_llm,
    )

    # --------------------------------------------------------
    # Local
    # --------------------------------------------------------

    if settings.use_local_llm:

        print(
            "[llm] provider = local TinyLlama"
        )

        # Direct Transformers implementation keeps the
        # standalone test simple and reliable.

        import torch

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        tokenizer = (
            _get_direct_local_tokenizer()
        )

        model = (
            _get_direct_local_model()
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Feedy, an AI customer "
                    "feedback classification assistant."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        device = next(
            model.parameters()
        ).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_ids = output_ids[
            0,
            input_length:
        ]

        result = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return result.strip()

    # --------------------------------------------------------
    # Remote
    # --------------------------------------------------------

    llm = get_chat_model()

    response = llm.invoke(
        prompt
    )

    if hasattr(
        response,
        "content",
    ):

        return str(
            response.content
        ).strip()

    return str(
        response
    ).strip()


# ============================================================
# DIRECT LOCAL MODEL CACHE
# ============================================================

@lru_cache(maxsize=1)
def _get_direct_local_tokenizer():

    from transformers import (
        AutoTokenizer,
    )

    print(
        "[llm] loading local tokenizer:",
        settings.hf_local_model,
    )

    return AutoTokenizer.from_pretrained(
        settings.hf_local_model
    )


@lru_cache(maxsize=1)
def _get_direct_local_model():

    import torch

    from transformers import (
        AutoModelForCausalLM,
    )

    print(
        "[llm] loading local TinyLlama model:",
        settings.hf_local_model,
    )

    if torch.cuda.is_available():

        model = (
            AutoModelForCausalLM.from_pretrained(
                settings.hf_local_model,
                dtype=torch.float16,
                device_map="auto",
            )
        )

    else:

        model = (
            AutoModelForCausalLM.from_pretrained(
                settings.hf_local_model,
                dtype=torch.float32,
            )
        )

        model.to("cpu")

    model.eval()

    return model


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def get_llm():
    """
    Compatibility helper.

    Existing code can use:

        get_llm().invoke(...)
    """

    return get_chat_model()