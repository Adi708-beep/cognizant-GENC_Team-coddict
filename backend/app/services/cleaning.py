import hashlib
import re
import unicodedata


# =========================================================
# BOILERPLATE
# =========================================================

BOILERPLATE_PATTERNS = [
    re.compile(
        r"^feedy\s*-\s*sample\s+customer\s+feedback\s+form$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^customer\s+feedback\s+form$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^demo\s+company\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^branch\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^date\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^form\s+no\.?\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^thank\s+you\s+for",
        re.IGNORECASE,
    ),
    re.compile(
        r"^confidential",
        re.IGNORECASE,
    ),
    re.compile(
        r"^store\s+copy",
        re.IGNORECASE,
    ),
    re.compile(
        r"^signature\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^contact\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^phone\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^email\s*:",
        re.IGNORECASE,
    ),
]


# =========================================================
# RATING LINES
# =========================================================

RATING_LINE = re.compile(
    r"^(?:"
    r"service"
    r"|staff\s+friendliness"
    r"|waiting\s+time"
    r"|product\s+availability"
    r")\s+\d+\s*/\s*\d+$",
    re.IGNORECASE,
)


# =========================================================
# QUESTION MARKER
# =========================================================

QUESTION_MARKER = re.compile(
    r"^\s*Q\s*\d+[\).:]",
    re.IGNORECASE,
)


# =========================================================
# QUESTION + ANSWER
# =========================================================
#
# Supports all of these:
#
#   How was the service?: It was good.
#   How was the service?: It was good.
#   Please share comments.: It was good.
#   Please share comments: It was good.
#
# The separator is therefore allowed to be:
#
#   ?
#   .
#   nothing
#
# followed by :
#

QUESTION_ANSWER_PATTERN = re.compile(
    r"^"
    r"(.*?)"
    r"(?:\?|(?<!\w)\.)?"
    r"\s*:\s*"
    r"(.*)"
    r"$",
    re.DOTALL,
)


# =========================================================
# NORMALISE
# =========================================================

def normalise(text: str) -> str:
    """
    Normalize extracted text while preserving useful structure.

    Operations:
    - Unicode normalization
    - smart quote normalization
    - dash normalization
    - repair hyphenated line breaks
    - collapse spaces and tabs
    - preserve meaningful newlines
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = (
        text
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )

    # Repair words split by a PDF line break.
    #
    # Example:
    #
    #   feed-
    #   back
    #
    # becomes:
    #
    #   feedback
    #
    text = re.sub(
        r"-\n(?=\w)",
        "",
        text,
    )

    # Collapse spaces/tabs but preserve newlines.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove spaces around newlines.
    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    # Prevent excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# BOILERPLATE DETECTION
# =========================================================

def _is_boilerplate(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    for pattern in BOILERPLATE_PATTERNS:
        if pattern.match(line):
            return True

    return False


# =========================================================
# NOISE DETECTION
# =========================================================

def _is_noise(line: str) -> bool:
    line = line.strip()

    if not line:
        return True

    if len(line) < 3:
        return True

    if _is_boilerplate(line):
        return True

    if line.lower() == "ratings":
        return True

    if RATING_LINE.match(line):
        return True

    return False


# =========================================================
# CLEAN ANSWER
# =========================================================

def _clean_answer(answer: str) -> str:
    """
    Clean an extracted answer.

    PDF extraction commonly wraps one sentence across
    several physical lines, so those lines are joined.
    """

    if not answer:
        return ""

    lines = answer.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Ratings section marks the end of feedback.
        if re.match(
            r"^ratings\b",
            line,
            re.IGNORECASE,
        ):
            break

        # Remove rating rows if they appear inside the block.
        if RATING_LINE.match(line):
            continue

        # Remove boilerplate.
        if _is_boilerplate(line):
            continue

        cleaned_lines.append(line)

    # Join PDF-wrapped lines.
    answer = " ".join(cleaned_lines)

    # Normalize whitespace.
    answer = re.sub(
        r"\s+",
        " ",
        answer,
    ).strip()

    return answer


# =========================================================
# CLEAN QUESTION
# =========================================================

def _clean_question(question: str) -> str:
    """
    Normalize the question text.

    The trailing question punctuation is normalized so
    different source forms produce consistent output.
    """

    question = re.sub(
        r"\s+",
        " ",
        question,
    ).strip()

    # Remove trailing punctuation that belonged to the
    # question/colon separator.
    question = re.sub(
        r"[.:?]+\s*$",
        "",
        question,
    ).strip()

    return question


# =========================================================
# EXTRACT QUESTION BLOCKS
# =========================================================

def _extract_question_blocks(text: str) -> list[str]:
    """
    Find every Q-numbered feedback block.

    Each block continues until:

        - the next Q-number
        - Ratings
        - end of document

    This works with answers wrapped across multiple
    physical PDF lines.
    """

    pattern = re.compile(
        r"""
        ^
        \s*
        Q\s*\d+[\).:]
        .*?
        (?=
            ^\s*Q\s*\d+[\).:]
            |
            ^\s*Ratings\b
            |
            \Z
        )
        """,
        re.IGNORECASE
        | re.MULTILINE
        | re.DOTALL
        | re.VERBOSE,
    )

    return [
        match.group(0).strip()
        for match in pattern.finditer(text)
    ]


# =========================================================
# PARSE QUESTION BLOCK
# =========================================================

def _parse_question_block(
    block: str,
) -> str | None:
    """
    Convert one question block into:

        Question: Answer

    Supports:

        Question?: Answer
        Question.: Answer
        Question: Answer
    """

    if not block:
        return None

    # Remove Q-number prefix.
    block = re.sub(
        r"^\s*Q\s*\d+[\).:]\s*",
        "",
        block,
        count=1,
        flags=re.IGNORECASE,
    )

    # Flatten physical PDF lines.
    block = re.sub(
        r"\s+",
        " ",
        block,
    ).strip()

    match = QUESTION_ANSWER_PATTERN.match(
        block
    )

    if not match:
        return None

    question = _clean_question(
        match.group(1)
    )

    answer = _clean_answer(
        match.group(2)
    )

    if not question:
        return None

    if not answer:
        return None

    # Ignore extremely short answers.
    if len(answer.split()) < 4:
        return None

    return f"{question}: {answer}"


# =========================================================
# SPLIT RESPONSES
# =========================================================

def split_responses(
    text: str,
) -> list[str]:
    """
    Extract individual customer feedback responses.

    Example input:

        Q1. How was the service?: Good.
        Q2. What did you like?: Staff.
        Q3. What can improve?: Waiting time.
        Q4. Additional comments.: Very good.
        Ratings
        ...

    Returns:

        [
            "How was the service: Good.",
            "What did you like: Staff.",
            "What can improve: Waiting time.",
            "Additional comments: Very good."
        ]

    Metadata, ratings, signatures, and boilerplate
    are excluded.
    """

    text = normalise(text)

    if not text:
        return []

    # -----------------------------------------------------
    # PRIMARY PARSER
    # -----------------------------------------------------

    blocks = _extract_question_blocks(
        text
    )

    responses = []

    for block in blocks:

        response = _parse_question_block(
            block
        )

        if response:
            responses.append(response)

    # If Q-numbered blocks exist, use only those.
    #
    # This is important because falling back to generic
    # paragraph parsing could turn the entire PDF into one
    # response.
    if responses:
        return dedupe(responses)

    # -----------------------------------------------------
    # GENERIC FALLBACK
    # -----------------------------------------------------

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if _is_noise(paragraph):
            continue

        lines = paragraph.splitlines()

        useful_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if _is_noise(line):
                continue

            useful_lines.append(line)

        paragraph = " ".join(
            useful_lines
        )

        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph,
        ).strip()

        if len(paragraph.split()) >= 6:
            responses.append(
                paragraph
            )

    # -----------------------------------------------------
    # LAST RESORT
    # -----------------------------------------------------

    if (
        not responses
        and len(text.split()) >= 4
    ):
        cleaned_lines = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            if _is_noise(line):
                continue

            cleaned_lines.append(line)

        fallback = " ".join(
            cleaned_lines
        )

        fallback = re.sub(
            r"\s+",
            " ",
            fallback,
        ).strip()

        if fallback:
            responses.append(
                fallback
            )

    return dedupe(responses)


# =========================================================
# DEDUPLICATION
# =========================================================

def dedupe(
    items: list[str],
) -> list[str]:
    """
    Remove duplicate responses while preserving order.
    """

    seen = set()

    output = []

    for item in items:

        key = re.sub(
            r"[^a-z0-9]+",
            "",
            item.lower(),
        )

        if key and key not in seen:

            seen.add(key)

            output.append(item)

    return output


# =========================================================
# SHA-1 HASH
# =========================================================

def text_hash(
    text: str,
) -> str:
    """
    Generate a stable SHA-1 hash.

    Used for response deduplication and record identity.
    """

    key = re.sub(
        r"[^a-z0-9]+",
        "",
        text.lower(),
    )

    return hashlib.sha1(
        key.encode("utf-8")
    ).hexdigest()