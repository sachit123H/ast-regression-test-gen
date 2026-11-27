import re
import pypdf

reader = pypdf.PdfReader("docs/thesis/thesis.pdf")
print("Num pages:", len(reader.pages))
full_text = ""
for i, page in enumerate(reader.pages):
    t = page.extract_text() or ""
    full_text += f"\n--- Page {i+1} ---\n" + t

# Filter out known string literal in Lang-17 JUnit trace
text_for_cites = full_text.replace("expected:<A?[?]>", "")
q_cites = re.findall(r"\[\?\]", text_for_cites)
q_refs = re.findall(r"\?\?", full_text)
print("Undefined citations [?]:", len(q_cites))
print("Undefined references ??:", len(q_refs))

# Verify essential updated sections & terms, and ensure deprecated TOST claim is absent
for term in [
    "65",
    "32.3%",
    "18.5%",
    "41.5%",
    "24.6%",
    "Strict Assertion",
    "pilot",
    "Math-45",
    "Math-5",
    "Lang-17",
    "HyRep-BRT",
    "Multi-SWE-bench",
    "Table 1",
]:
    print(f'Found "{term}":', term in full_text)

print('Deprecated "TOST" absent:', "TOST" not in full_text)
