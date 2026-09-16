def parse_message(message: Any, parser: PydanticOutputParser) -> BaseModel:
    """Turn a model reply into a validated object, tolerating fences and prose."""
    text = message.content if isinstance(message, BaseMessage) else str(message)
    if isinstance(text, list):
        # some providers return a list of content blocks rather than a string
        text = "".join(
            part.get("text", "") for part in text if isinstance(part, dict)
        )

    try:
        return parser.parse(text)
    except Exception:  # noqa: BLE001 - fall through to the tolerant path
        pass

    blob = first_json_object(text)
    if blob is None:
        raise LLMError(f"model did not return JSON: {text[:200]!r}")
    try:
        return parser.pydantic_object.model_validate(blob)
    except ValidationError as exc:
        raise LLMError(
            f"JSON did not match {parser.pydantic_object.__name__}: {exc}"
        ) from exc


def first_json_object(text: str) -> dict | None:
    """Pull the first complete JSON object out of a reply.

    Brace-counting, so a trailing sentence after the closing brace does not
    break parsing.
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
                    return json.loads(cleaned[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


# vision tier
def vision_read(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Tier-3 OCR: hand the page image to a vision-capable chat model."""
    if not settings.gemini_api_key:
        raise LLMError("vision OCR requested but GEMINI_API_KEY is not set")
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
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded}"},
        ]
    )
    try:
        return str(model.invoke([message]).content).strip()
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"vision model failed: {exc}") from exc