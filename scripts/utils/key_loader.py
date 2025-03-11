import json
import os

def load_api_keys():
    # Risali alla cartella principale del progetto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("scripts","")
    keys_path = os.path.join(base_dir, "credentials", "api_keys.json")

    try:
        with open(keys_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Errore: api_keys.json non trovato!")
        return {}
    except json.JSONDecodeError:
        print("Errore: api_keys.json non è un JSON valido!")
        return {}
