from pathlib import Path

from app.services.extraction import extract_pdf
from app.services.cleaning import normalise, split_responses, text_hash


pdf_path = Path("feedy_sample_feedback.pdf")

# Step 1: Extract
result = extract_pdf(pdf_path.read_bytes())

print("=== EXTRACTION ===")
print("METHOD:", result.method)
print("CONFIDENCE:", result.confidence)
print("PAGES:", result.pages)

# Step 2: Normalize
clean_text = normalise(result.text)

print("\n=== NORMALISED TEXT ===")
print(clean_text)

# Step 3: Split into responses
responses = split_responses(clean_text)

print("\n=== RESPONSES ===")

for i, response in enumerate(responses, start=1):
    print(f"\n{i}. {response}")

# Step 4: Generate hashes
print("\n=== HASHES ===")

for i, response in enumerate(responses, start=1):
    print(f"{i}. {text_hash(response)}")