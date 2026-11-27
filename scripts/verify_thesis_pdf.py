import pypdf
import re

reader = pypdf.PdfReader('docs/thesis/thesis.pdf')
print('Num pages:', len(reader.pages))
full_text = ''
for i, page in enumerate(reader.pages):
    t = page.extract_text() or ''
    full_text += f'\n--- Page {i+1} ---\n' + t

# Filter out known string literal in Lang-17 JUnit trace
text_for_cites = full_text.replace('expected:<A?[?]>', '')
q_cites = re.findall(r'\[\?\]', text_for_cites)
q_refs = re.findall(r'\?\?', full_text)
print('Undefined citations [?]:', len(q_cites))
print('Undefined references ??:', len(q_refs))

# Check for presence of key numbers and sections
for term in ['65', '41.5%', '24.6%', '0.5144', '0.5148', '67.3%', 'Math-45', 'Math-5', 'Lang-17', 'HyRep-BRT', 'Table 1']:
    print(f'Found "{term}":', term in full_text)
