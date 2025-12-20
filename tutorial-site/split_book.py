
import os

input_file = r'c:\Users\abish\OneDrive\Documents\Side Projects\Book\DS Book1.txt'
output_dir = r'c:\Users\abish\OneDrive\Documents\Side Projects\Book\tutorial-site\docs\chapters'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

boundaries = [
    (52, "Chapter 1: The Data Science Landscape"),
    (290, "Chapter 2: Python Essentials for Data Analysis"),
    (726, "Chapter 3: Mastering Data Manipulation with Pandas"),
    (1170, "Chapter 4: Data Cleaning and Preparation"),
    (1541, "Chapter 5: Exploratory Data Analysis and Visualization"),
    (1950, "Chapter 6: Statistical Foundations for Decision Making"),
    (2270, "Chapter 7: Predictive Modeling with Linear Regression"),
    (2698, "Chapter 8: Classification Algorithms for Categorical Outcomes"),
    (3105, "Chapter 9: Unsupervised Learning and Pattern Discovery"),
    (3459, "Chapter 10: The Capstone Project: End-to-End Workflow")
]

def clean_line(line):
    # Remove prefix like "52: " if it exists from previous tools, 
    # but the raw file doesn't have it.
    return line

# Process Intro (Lines 1 to 51)
intro_content = lines[0:51]
with open(os.path.join(output_dir, 'intro.md'), 'w', encoding='utf-8') as f:
    f.write("# Introduction\n\n")
    for line in intro_content:
        f.write(line)

# Process Chapters
for i in range(len(boundaries)):
    start_ln, title = boundaries[i]
    end_ln = boundaries[i+1][0] if i+1 < len(boundaries) else len(lines) + 1
    
    filename = f"chapter-{i+1}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Slice lines (boundaries are 1-indexed in my list from previous tool, so adjust)
    chapter_lines = lines[start_ln-1 : end_ln-1]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Check if the first line is the chapter title and skip it if redundant
        # but I'll force my clean title
        f.write(f"# {title}\n\n")
        
        in_code_block = False
        
        for j, line in enumerate(chapter_lines):
            # Skip the first line if it's the title we just manually added
            if j == 0 and ('Chapter' in line):
                continue
                
            stripped = line.strip()
            
            # Heuristic for code blocks
            if stripped in ['python', 'text', 'javascript', 'html', 'css', 'sql', 'bash']:
                if in_code_block:
                    f.write("```\n\n")
                f.write(f"```{stripped}\n")
                in_code_block = True
                continue
            
            # If line is 'Output:', and next line is 'text', it's a code block transition
            if stripped == 'Output:' and j+1 < len(chapter_lines) and chapter_lines[j+1].strip() == 'text':
                 if in_code_block:
                    f.write("```\n\n")
                 f.write("**Output:**\n\n")
                 # The 'text' marker will be handled by the next iteration
                 continue

            # Heuristic to close a code block: 
            # If we see a line that looks like a sub-header or significant text after a blank line.
            # But the book format is a bit loose.
            # I'll look for lines that are just text without common code patterns.
            # For now, I'll only close on another block marker or end of file.
            # UNLESS it's a clear paragraph.
            
            # Let's try to detect natural language vs code.
            # If it's more than 3 words and doesn't start with code symbols.
            if in_code_block and stripped and not any(stripped.startswith(c) for c in ['#', 'import ', 'from ', 'def ', 'print(', 'plt.', 'df[', 'df.', 'sns.', 'x =', 'y =', 'plt.show()']):
                # If the line starts with a capital letter and has multiple words, it might be text.
                # But Python can have `Total = 1`.
                # If there's a space before the first word, it's likely code.
                if not line.startswith(' ') and not line.startswith('\t') and len(stripped.split()) > 3 and stripped[0].isupper():
                    f.write("```\n\n")
                    in_code_block = False
            
            f.write(line)
            
        if in_code_block:
            f.write("```\n")

print("Splitting complete.")
