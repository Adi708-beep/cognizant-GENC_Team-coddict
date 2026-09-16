from pathlib import Path

from app.services.extraction import extract_pdf


pdf_path = Path("feedy_sample_feedback.pdf")

result = extract_pdf(
    pdf_path.read_bytes()
)

print("METHOD:", result.method)
print("CONFIDENCE:", result.confidence)
print("PAGES:", result.pages)
print("NOTE:", result.note)

print("\nTEXT:")
print(result.text)