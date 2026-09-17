from app.services.llm import generate


print("=" * 60)
print("FEEDY LOCAL TINYLLAMA TEST")
print("=" * 60)

prompt = """
Classify this customer feedback.

Feedback:
The staff were very helpful and friendly.

Choose exactly one category:
Excellent
Good
Need Improvements
Poor

Return ONLY JSON in this format:

{
  "category": "Excellent",
  "confidence": 0.90,
  "rationale": "The customer was very positive about the staff."
}
"""

print()
print("[test] sending prompt to local TinyLlama...")

try:
    result = generate(
        prompt,
        max_new_tokens=150,
    )

    print()
    print("=" * 60)
    print("MODEL OUTPUT")
    print("=" * 60)
    print(result)
    print("=" * 60)

except Exception as exc:
    print()
    print("=" * 60)
    print("LOCAL MODEL TEST FAILED")
    print("=" * 60)
    print(type(exc).__name__)
    print(str(exc))
    print("=" * 60)