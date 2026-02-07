import re
import json
import os

def parse_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by back to top link or horizontal rule which often separates questions
    # But a more reliable way is to find the ### headings
    sections = re.split(r'\n### ', content)
    
    # The first section is usually the TOC, skip it
    questions_data = []
    
    for section in sections[1:]:
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        question_text = lines[0].strip()
        options = []
        correct_answers = []
        image_path = None
        
        # Parse the rest of the lines for options, images, etc.
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            # Match image: ![Question 1](images/question1.jpg)
            img_match = re.search(r'!\[.*?\]\((.*?)\)', line)
            if img_match:
                image_path = img_match.group(1)
                continue
                
            # Match options: - [ ] Option or - [x] Correct Option
            option_match = re.match(r'- \[( |x)\] (.*)', line)
            if option_match:
                is_correct = option_match.group(1) == 'x'
                text = option_match.group(2).strip()
                options.append(text)
                if is_correct:
                    # Store indices (0-based)
                    correct_answers.append(len(options) - 1)
                continue
            
            # If line starts with bold or is just text and we don't have enough options yet, 
            # might be part of the question text or a note
            if line.startswith('**[⬆ Back to Top]'):
                break
                
            if not options and not image_path:
                # Still part of the question text possibly?
                question_text += " " + line

        if question_text and options:
            # Determine if it's a multiple choice (multiple answers)
            is_multiple = len(correct_answers) > 1 or "Choose two" in question_text or "Choose three" in question_text
            
            questions_data.append({
                "id": len(questions_data) + 1,
                "question": question_text,
                "options": options,
                "answer": correct_answers,
                "image": image_path,
                "multiple": is_multiple
            })
            
    return questions_data

if __name__ == "__main__":
    input_file = r"c:\Users\abish\OneDrive\Documents\Side Projects\AWS_Exam\questions.md"
    output_file = r"c:\Users\abish\OneDrive\Documents\Side Projects\AWS_Exam\questions.json"
    
    if os.path.exists(input_file):
        print(f"Parsing {input_file}...")
        data = parse_questions(input_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully parsed {len(data)} questions into {output_file}")
    else:
        print(f"Error: {input_file} not found.")
