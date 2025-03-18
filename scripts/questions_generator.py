import json
import os
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import llms

def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def take_random_text(max_words, text):
    """Extracts a random portion of the text with a max word limit."""
    words = text.split()
    if len(words) <= max_words:
        return text
    start = random.randint(0, len(words) - max_words)
    return " ".join(words[start:start + max_words])

def load_texts(texts_dir):
    """Loads text files from a directory."""
    texts = []
    for file_name in tqdm(os.listdir(texts_dir), desc="Loading .txt files"):
        if file_name.endswith(".txt"):
            file_path = os.path.join(texts_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append({"text": f.read(), "id": file_name})
            except UnicodeDecodeError:
                # If there is a decoding error, try an alternative encoding
                with open(file_path, 'r', encoding='ISO-8859-1') as f:
                    texts.append({"text": f.read(), "id": file_name})
    return texts

def generate_questions(base_prompt, texts, n_questions, max_words, provider, model_name, questions_file_path, existing_questions=None):
    """Generates multiple-choice questions from given texts and saves every 20 questions."""
    if existing_questions is None:
        existing_questions = []

    generated_questions = existing_questions
    seen_questions = {q['question'] for q in existing_questions}
    attempts = 0
    max_attempts = 100
    
    while len(generated_questions) < n_questions and attempts < max_attempts:
        article = random.choice(texts)
        text_to_use = take_random_text(max_words, article['text']) if max_words else article['text']
        prompt = base_prompt.replace("<<<text>>>", text_to_use)
        
        try:
            question_data = llms.generate_question(provider, model_name, prompt)
            question_data['id'] = article['id']
            
            # If the question has not been seen before, add it
            if question_data['question'] not in seen_questions:
                seen_questions.add(question_data['question'])
                random.shuffle(question_data['options'])
                generated_questions.append(question_data)
                print(f"{len(generated_questions)}/{n_questions} questions generated.")

                attempts = 0
                
                # Save the questions every 20 questions
                if len(generated_questions) % 20 == 0:
                    with open(questions_file_path, "w", encoding="utf-8") as json_file:
                        json.dump(generated_questions, json_file, indent=4, ensure_ascii=False)
                    print(f"Saved {len(generated_questions)} questions to {questions_file_path}.")
            else:
                attempts += 1
        except Exception as e:
            attempts += 1
            print(f"Error: {e}")
            if attempts == max_attempts:
                print(f"{len(generated_questions)} questions generated. Process interrupted due to too many errors.")
                break
    

    with open(questions_file_path, "w", encoding="utf-8") as json_file:
        json.dump(generated_questions, json_file, indent=4, ensure_ascii=False)
    
    return generated_questions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--questions_path", type=str, default="questions", help="Folder for questions")
    parser.add_argument("--name_file_questions", type=str, default="generated_questions.json", help="Output JSON file")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (e.g openai, gemini, groq)")
    parser.add_argument("--model_name", type=str, default="gpt4o_LowFilter", help="The specific model name")
    parser.add_argument("--prompt_path", type=str, default="scripts/utils/base_prompt.txt", help="Base prompt file")
    parser.add_argument("--texts_path", type=str, default="texts", help="Folder for text files")
    parser.add_argument("--n_questions", type=int, default=500, help="Number of questions to generate")
    parser.add_argument("--max_words_per_q", type=int, default=None, help="Max words per question context")
    parser.add_argument("--skip_existing", type=bool, default=True, nargs="?", const=True, help="Skip processing if passages.json already exists")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    texts_dir = dataset_path / args.texts_path
    questions_dir = dataset_path / args.questions_path
    questions_file_path = questions_dir / args.name_file_questions
    prompt_file_path = args.prompt_path

    ensure_dir(questions_dir)
    
    # Read the base prompt
    with open(prompt_file_path, encoding="utf-8") as f:
        base_prompt = f.read()
    
    # Load the texts
    texts = load_texts(texts_dir)
    
    # Load existing questions if the file exists
    existing_questions = []
    if questions_file_path.exists() and args.skip_existing:
        with open(questions_file_path, "r", encoding="utf-8") as json_file:
            existing_questions = json.load(json_file)
    
    # If the file already contains the required number of questions, skip generation
    if len(existing_questions) >= args.n_questions:
        print(f"File already contains {len(existing_questions)} questions. Skipping generation.")
    else:
        # If there are fewer questions than required, generate the missing ones
        print(f"Generating {args.n_questions} questions.")
        generate_questions(base_prompt, texts, args.n_questions, args.max_words_per_q, args.provider, args.model_name, questions_file_path, existing_questions)
        print(f"Questions generated and saved to {questions_file_path}")

if __name__ == "__main__":
    main()
