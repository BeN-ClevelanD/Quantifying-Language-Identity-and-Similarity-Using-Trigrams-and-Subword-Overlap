import os
import re
from collections import defaultdict
#The below script checks validation files for non-basic characters and reports their presence. It is to confirm strategy for text normalisation.
VAL_DIR = "data"
VAL_FILES = [f for f in os.listdir(VAL_DIR) if f.startswith("val.") and f.endswith(".txt")]
BASIC_CHARS = set("abcdefghijklmnopqrstuvwxyz 0")
char_occurrences = defaultdict(set)

for filename in VAL_FILES:
    with open(os.path.join(VAL_DIR, filename), 'r', encoding='utf-8') as f:
        for line in f:
            for char in line.strip():
                if char not in BASIC_CHARS:
                    char_occurrences[char].add(filename)

all_found_chars = sorted(char_occurrences.keys())
print("\n Non-basic character presence in validation files:\n")
for char in all_found_chars:
    files = list(char_occurrences[char])
    print(f"Char: {repr(char):>6} | Found: True | Files: {', '.join(files)}")
UNUSUAL_EXPECTED_CHARS = set([
    '#', '_', '%', '$', '@', '!', '.', ',', ':', ';', '?', '"', "'", '-', '(', ')',
    'ß', 'æ', 'ı', 'ł', 'ʾ', 'ʼ', 'α', 'δ', 'ε', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν',
    'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'χ', 'ω',
    '𓅓', '𓆎', '𓊖', '𓏏', 'さ', 'な', 'イ', 'キ', 'ク', 'ケ', 'ッ', 'ハ', 'ヒ', 'ン'
])
print("\nExpected problematic characters NOT found in any file:\n")
not_found = UNUSUAL_EXPECTED_CHARS - set(all_found_chars)
for ch in sorted(not_found):
    print(f"Char: {repr(ch):>6} | Found: False")
