import io
from dataclasses import dataclass

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
from pytesseract import Output

from app.config import settings
from app.services.llm import vision_read

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

MIN_CHARS_PER_PAGE = 40    # below this, assume there is no usable text layer
MIN_OCR_CONFIDENCE = 0.55  # below this, escalate to the vision tier


@dataclass
class Extraction:
    text: str
    method: str    # digital | ocr | vision | spreadsheet
    confidence: float  # 0.0 - 1.0, honest estimate of extraction quality
    pages: int = 1
    note: str = ""


# tier 1
def _pdf_text_layer(file_bytes: bytes) -> tuple[str, int]:
    out, pages = [], 0
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return "\n".join(out).strip(), pages


# tier 2
def _ocr_image(image: Image.Image) -> tuple[str, float]:
    """OCR one page image, returning text plus Tesseract's mean word confidence."""
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words, confs = [], []
    for token, conf in zip(data["text"], data["conf"]):
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue
        if token.strip() and conf_value >= 0:
            words.append(token)
            confs.append(conf_value)
    text = " ".join(words)
    confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return text, round(confidence, 3)


def _pdf_render_pages(file_bytes: bytes) -> list[Image.Image]:
    images = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=settings.ocr_dpi)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    return images


def _pdf_ocr(file_bytes: bytes) -> Extraction:
    images = _pdf_render_pages(file_bytes)
    texts, confs = [], []
    for image in images:
        text, conf = _ocr_image(image)
        texts.append(text)
        confs.append(conf)
    mean_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
    return Extraction("\n".join(texts).strip(), "ocr", mean_conf, len(images))


# tier 3
def _vision_pdf(file_bytes: bytes) -> Extraction:
    images = _pdf_render_pages(file_bytes)
    texts = []
    for image in images[:4]:  # cap: vision calls are the expensive tier
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        texts.append(vision_read(buffer.getvalue(), "image/png"))
    return Extraction("\n".join(texts).strip(), "vision", 0.75, len(images), "read by vision model")


# dispatcher
def extract_pdf(file_bytes: bytes) -> Extraction:
    text, pages = _pdf_text_layer(file_bytes)
    if pages and len(text) >= MIN_CHARS_PER_PAGE * pages:
        return Extraction(text, "digital", 1.0, pages)

    ocr = _pdf_ocr(file_bytes)
    good_enough = (ocr.confidence >= MIN_OCR_CONFIDENCE
                   and len(ocr.text) >= MIN_CHARS_PER_PAGE)
    if good_enough or not settings.enable_vision_ocr:
        ocr.note = "no text layer; OCR used"
        return ocr

    try:
        return _vision_pdf(file_bytes)
    except Exception as exc:  # noqa: BLE001
        ocr.note = f"vision tier unavailable ({exc}); OCR result kept"
        return ocr


def extract_image(file_bytes: bytes, mime_type: str = "image/png") -> Extraction:
    image = Image.open(io.BytesIO(file_bytes))
    text, conf = _ocr_image(image)
    if conf >= MIN_OCR_CONFIDENCE and len(text) >= MIN_CHARS_PER_PAGE:
        return Extraction(text, "ocr", conf, 1)

    if settings.enable_vision_ocr:
        try:
            return Extraction(vision_read(file_bytes, mime_type), "vision", 0.75, 1, "low OCR confidence; vision used")
        except Exception as exc:  # noqa: BLE001
            return Extraction(text, "ocr", conf, 1, f"vision failed ({exc})")
    return Extraction(text, "ocr", conf, 1, "low OCR confidence")


def read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """CSV or Excel to a DataFrame. Schema understanding happens elsewhere."""
    name = filename.lower()
    if name.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(file_bytes))
    elif name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported table format. Use CSV or Excel.")
    frame = frame.dropna(how="all")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame