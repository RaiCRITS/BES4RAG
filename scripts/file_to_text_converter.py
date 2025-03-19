import os
import json
import argparse
from pathlib import Path
from shutil import copy2
from PyPDF2 import PdfReader

def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def create_description_file(filepath, descriptions_dir):
    """Create a description file for the original PDF, JSON, or TXT."""
    filename = os.path.basename(filepath).replace(filepath.suffix, ".txt")
    description_path = os.path.join(descriptions_dir, filename)
    
    if not os.path.exists(description_path):  # Check if description already exists
        with open(description_path, "w", encoding="utf-8") as desc_file:
            desc_file.write(f"Description for {filename} based on the original content.\n")
        print(f"Created description file: {description_path}")

def process_json(filepath, texts_dir, mapping, skip_existing):
    """Process a JSON file and convert its content into text files."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            filename = os.path.basename(filepath).replace(".json", "")
            if isinstance(data, list):
                for i, el in enumerate(data):
                    for j, (key, value) in enumerate(el.items()):
                        text_filename = f"{filename}___{i}_{j}.txt"
                        text_path = os.path.join(texts_dir, text_filename)
                        if not skip_existing or not os.path.exists(text_path):  # Check if file exists
                            text_content = f"{key}: {value}"
                            with open(text_path, "w", encoding="utf-8") as text_file:
                                text_file.write(text_content)
                        mapping["texts_to_files"][text_filename] = str(filepath)
            else:
                for i, (key, value) in enumerate(data.items()):
                    text_filename = f"{filename}___{i}.txt"
                    text_path = os.path.join(texts_dir, text_filename)
                    if not skip_existing or not os.path.exists(text_path):  # Check if file exists
                        text_content = f"{key}: {value}"
                        with open(text_path, "w", encoding="utf-8") as text_file:
                            text_file.write(text_content)
                    mapping["texts_to_files"][text_filename] = str(filepath)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {filepath}")

def process_txt(filepath, texts_dir, mapping, skip_existing):
    """Copy a text file to the target directory."""
    dest_path = os.path.join(texts_dir, os.path.basename(filepath))
    text_filename = os.path.basename(filepath)
    if not skip_existing or not os.path.exists(dest_path):  # Check if file exists
        copy2(filepath, dest_path)
    mapping["texts_to_files"][text_filename] = str(filepath)

def process_pdf(filepath, texts_dir, mapping, skip_existing):
    """Convert each page of a PDF into a separate text file."""
    pdf_reader = PdfReader(filepath)
    filename = os.path.basename(filepath).replace(".pdf", "")
    for i, page in enumerate(pdf_reader.pages):
        text_filename = f"{filename}_page_{i}.txt"
        text_path = os.path.join(texts_dir, text_filename)
        if not skip_existing or not os.path.exists(text_path):  # Check if file exists
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(page.extract_text() or "")
        mapping["texts_to_files"][text_filename] = str(filepath)

def main():
    """Main function to process files in the dataset directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--files_path", type=str, default="files", help="Name of the folder containing the files")
    parser.add_argument("--texts_path", type=str, default="texts", help="Name of the folder where converted files will be stored")
    parser.add_argument("--descriptions_path", type=str, default="descriptions", help="Name of the folder where descriptions will be stored")
    parser.add_argument("--skip_existing", type=bool, default=True, nargs="?", const=True, help="Skip processing if passages.json already exists")
    parser.add_argument("--create_descriptions", type=bool, default=False, nargs="?", const=True, help="Create descriptions automatically for each file")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    files_dir = dataset_path / args.files_path
    texts_dir = dataset_path / args.texts_path
    descriptions_dir = dataset_path / args.descriptions_path
    mapping_dir = dataset_path / "mapping"
    mapping_file = os.path.join(mapping_dir,"file_mapping.json")
    
    # Check if the descriptions directory exists, and if so, disable automatic description creation
    if descriptions_dir.exists():
        print(f"Descriptions directory {descriptions_dir} already exists. Automatic description creation will be disabled.")
        args.create_descriptions = False
    
    ensure_dir(texts_dir)
    ensure_dir(descriptions_dir)
    ensure_dir(mapping_dir)
    
    if os.path.exists(mapping_file) and args.skip_existing:
        print(f"Mapping already exists at {mapping_file}. No action taken.")
        return
    
    mapping = {"texts_to_files": {}}
    for filepath in files_dir.glob("*"):
        if filepath.suffix == ".json":
            process_json(filepath, texts_dir, mapping, args.skip_existing)
        elif filepath.suffix == ".txt":
            process_txt(filepath, texts_dir, mapping, args.skip_existing)
        elif filepath.suffix == ".pdf":
            process_pdf(filepath, texts_dir, mapping, args.skip_existing)
        
        # If the 'create_descriptions' flag is set, create description files
        if args.create_descriptions:
            create_description_file(filepath, descriptions_dir)
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    
    print(f"Processing completed. Mapping saved in {mapping_file}")

if __name__ == "__main__":
    main()


