import os
import re
import unicodedata
#THis python program implements our text normalisation strategy for Wikipedia-style data (training data). Explanation of steps shown in report.
def remove_accents(content):
    """
    Strip diacritic marks from text using Unicode decomposition.
    """
    return ''.join(
        letter for letter in unicodedata.normalize('NFKD', content)
        if not unicodedata.combining(letter)
    )

def strip_non_basic_ascii(text):
    """
    Keep only lowercase a–z, digit 0, and space.
    """
    return ''.join(c for c in text if c in 'abcdefghijklmnopqrstuvwxyz 0')

def clean_line(raw_line):
    """
    Apply standardised transformation rules:
    - Convert to lowercase
    - Remove diacritics
    - Replace all digits with '0'
    - Remove all non-basic ASCII (a–z, 0, space)
    - Standardise spacing
    """
    result = raw_line.lower()
    result = remove_accents(result)
    result = re.sub(r'\d', '0', result)
    result = re.sub(r'[^\w\s]', '', result)
    result = re.sub(r'\s+', ' ', result)
    result = strip_non_basic_ascii(result)
    return result.strip()

def break_into_units(raw_block):
    """
    Basic splitting of paragraph blocks into sentence-like chunks using markers.
    """
    prepared = re.sub(r'([.!?])\s+', r'\1\n', raw_block)
    return prepared.splitlines()

def normalise_text_file(input_file, output_file):
    """
    Transform the original Wikipedia-style data into a clean format.
    """
    with open(input_file, 'r', encoding='utf-8') as reader:
        document = reader.read()

    fragments = break_into_units(document)

    transformed = [
        clean_line(frag) for frag in fragments
        if len(frag.strip()) > 0
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as writer:
        for entry in transformed:
            if len(entry) >= 3:
                writer.write(entry + '\n')

