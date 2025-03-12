import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse

# Import specific utilities from custom module
from utils import embeddings

def load_descriptions(descriptions_dir):
    """Check if descriptions_dir exists, then load descriptions.json or descriptions.txt."""
    if not os.path.exists(descriptions_dir):
        return None

    json_path = os.path.join(descriptions_dir, "descriptions.json")
    txt_path = os.path.join(descriptions_dir, "descriptions.txt")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    return None

def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def read_embeddings_models(file_path):
    """Read embeddings_models.csv or .xlsx and return a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")
    
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(file_path)
            df.columns = ["type", "model_name"]
        except:
            df = pd.read_csv(file_path,sep=";")
            df.columns = ["type", "model_name"]
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
        df.columns = ["type", "model_name"]
    else:
        raise ValueError("Unsupported format! Use .csv or .xlsx")
    
    
    return df.to_dict(orient="records")

def main():
    """Main function to process files in the dataset directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--embeddings_models_filename", type=str, default="embeddings_models.csv", help="File with embeddings models list")
    parser.add_argument("--texts_path", type=str, default="texts", help="Folder for converted files")
    parser.add_argument("--descriptions_path", type=str, default="descriptions", help="Folder for descriptions")
    parser.add_argument("--passages_path", type=str, default="passages", help="Folder for computed passages")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    texts_dir = dataset_path / args.texts_path
    descriptions_dir = dataset_path / args.descriptions_path
    passages_dir = dataset_path / args.passages_path
    embeddings_models_file = dataset_path / args.embeddings_models_filename
    mapping_file = dataset_path / "mapping/file_mapping.json"
    
    ensure_dir(passages_dir)
    descriptions = load_descriptions(descriptions_dir)
    embeddings_models = read_embeddings_models(embeddings_models_file)
    
    TOKENIZER_CACHE = {}
    for el in embeddings_models:
        embeddings.get_tokenizer(el, TOKENIZER_CACHE)
    
    passages = {"-".join([el['type'], el["model_name"].replace("/", "|")]): {} for el in embeddings_models}
    
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    for text_file in tqdm(mapping["texts_to_files"].keys()):

        try:
            with open(texts_dir / text_file, "r", encoding="utf-8") as file:
                text = file.read()
        except UnicodeDecodeError:
            # Se c'è un errore con utf-8, prova con una codifica alternativa
            with open(texts_dir / text_file, 'r', encoding='ISO-8859-1') as file:
                text = file.read()
        
        description = descriptions.get(mapping["texts_to_files"][text_file]) if isinstance(descriptions, dict) else descriptions
        
        for el in embeddings_models:
            name_model = "-".join([el['type'], el["model_name"].replace("/", "|")])
            passages[name_model][text_file] = embeddings.split_text(el, text, description=description, TOKENIZER_CACHE=TOKENIZER_CACHE, overlap=25)



    with open(os.path.join(passages_dir,"passages.json"), "w", encoding="utf-8") as f:
        json.dump(passages,f)

if __name__ == "__main__":
    main()
