import hashlib
import re
import unicodedata

BOILERPLATE = re.compile(
    r"^(customer\s+feedback\s+form|thank\s+you\s+for|please\s+rate|page\s+\d+|"
    r"confidential|form\s+no\.?|store\s+copy|branch\s*:|date\s*:|name\s*:|"
    r"signature\s*:|contact\s*:|phone\s*:|email\s*:)",
    re.IGNORECASE,
)

QUESTION_LINE = re.compile(
    r"^\s*(?:q\s*\d+[\).:]|\d+[\).]|[-*\u2022])?\s*"
    r"(.{0,80}?\?|comments?|remarks?|suggestions?|what\s+.{0,40}?)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)

RATING_ONLY = re.compile(r"^[\s\d/\.\-\*xX\u2713\u2714()\[\]]+$")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"-\n(\w)", r"\1", text)                       # de-hyphenate across line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return True
    if BOILERPLATE.match(stripped):
        return True
    if RATING_ONLY.match(stripped):
        return True
    return False


def split_responses(text: str) -> list[str]:
    """Turn one extracted form into a list of individual free-text answers.

    Strategy: prefer "Question: answer" pairs. If the form has none, fall back
    to paragraph blocks. Anything shorter than four words carries no
    classifiable sentiment and is dropped.
    """
    text = normalise(text)
    answers: list[str] = []

    for line in text.split("\n"):
        if _is_noise(line):
            continue
        match = QUESTION_LINE.match(line.strip())
        if match:
            question, answer = match.group(1).strip(), match.group(2).strip()
            if len(answer.split()) >= 4:
                answers.append(f"{question}: {answer}")
        elif len(line.split()) >= 8:
            answers.append(line.strip())

    if not answers:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text)]
        answers = [b for b in blocks if len(b.split()) >= 6 and not _is_noise(b)]

    if not answers and len(text.split()) >= 4:
        answers = [text]

    return dedupe(answers)


def dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = re.sub(r"[^a-z0-9]+", "", item.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def text_hash(text: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    return hashlib.sha1(key.encode()).hexdigest()