
import re

input_file = r'c:\Users\abish\OneDrive\Documents\Side Projects\Book\DS Book1.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'Chapter' in line:
            print(f"{i+1}: {line.strip()}")
