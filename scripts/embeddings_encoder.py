import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import pickle

# Import specific utilities from custom module
from utils import embeddings

def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def read_embeddings_models(file_path):
    """Read embeddings_models.csv or .xlsx and return a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")
    
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported format! Use .csv or .xlsx")
    
    df.columns = ["type", "model_name"]
    return df.to_dict(orient="records")

def main():
    """Main function to process files in the dataset directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--passages_path", type=str, default="passages", help="Folder for computed passages")
    parser.add_argument("--embeddings_models_filename", type=str, default="embeddings_models.csv", help="File with embeddings models list")
    parser.add_argument("--embeddings_path", type=str, default="embeddings", help="Folder for computed embeddings")

    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    passages_dir = dataset_path / args.passages_path
    embeddings_models_file = dataset_path / args.embeddings_models_filename
    embeddings_dir = dataset_path / args.embeddings_path

    ensure_dir(embeddings_dir)

    embeddings_models = read_embeddings_models(embeddings_models_file)
    
    TOKENIZER_CACHE = {}
    for el in embeddings_models:
        embeddings.get_tokenizer(el, TOKENIZER_CACHE)

    with open(passages_dir / "passages.json", "r", encoding="utf-8") as f:
        passages = json.load(f)

    for embedding in embeddings_models:
        name = "-".join([embedding['type'], embedding["model_name"].replace("/", "|")])
        passages_model = passages.get(name, {})
        transcripts = []
        
        for i, text_file in enumerate(passages_model.keys()):
            text_pass = passages_model[text_file]
            transcripts.append({"id": f"{text_file}#{i}", "text": text_pass})

        emb_file = embeddings.compute_embeddings(transcripts, embedding)

        with open(embeddings_dir / name, "wb") as f:
            pickle.dump(emb_file, f)

if __name__ == "__main__":
    main()

