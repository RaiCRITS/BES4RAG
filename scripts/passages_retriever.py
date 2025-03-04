import json
import os
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from utils import retrieval

def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def main():
    """Main function to process files in the dataset directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--questions_path", type=str, default="questions", help="Folder for converted questions")
    parser.add_argument("--name_file_questions", type=str, default="generated_questions.json", help="Name of JSON file containing questions")
    parser.add_argument("--embeddings_path", type=str, default="embeddings", help="Folder for computed embeddings")
    parser.add_argument("--retrieval_path", type=str, default="retrieval", help="Folder for retrieval results")
    
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    questions_dir = dataset_path / args.questions_path
    file_questions_path = questions_dir / args.name_file_questions
    embeddings_dir = dataset_path / args.embeddings_path
    retrieval_dir = dataset_path / args.retrieval_path
    
    ensure_dir(retrieval_dir)

    retrieval_questions_dir = retrieval_dir / file_questions_path.split("/")[-1].replace(".json","")

    ensure_dir(retrieval_questions_dir)
    
    with open(file_questions_path, "r", encoding="utf-8") as json_file:
        questions = json.load(json_file)
    
    embeddings_dict = retrieval.read_embeddings(embeddings_dir)
    
    for model_name, embeddings in embeddings_dict.items():
        print(model_name)
        safe_name = model_name.replace("/", "|")
        retrieval_dict = {}
        
        for question in tqdm([q["question"] for q in questions]):
            retrieval_dict[question] = retrieval.retrieve_passages(question, embeddings)
        
        with open(retrieval_questions_dir / safe_name, "wb") as f:
            pickle.dump(retrieval_dict, f)

if __name__ == "__main__":
    main()








