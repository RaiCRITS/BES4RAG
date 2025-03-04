import os
import json
import pickle
import argparse
import matplotlib.pyplot as plt
from pathlib import Path





def main():
    """Main function to process files in the dataset directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--questions_path", type=str, default="questions", help="Folder for converted questions")
    parser.add_argument("--name_file_questions", type=str, default="generated_questions.json", help="Name of JSON file containing questions")
    parser.add_argument("--retrieval_path", type=str, default="retrieval", help="Folder for retrieval results")



    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    questions_dir = dataset_path / args.questions_path
    file_questions_path = questions_dir / args.name_file_questions
    retrieval_dir = dataset_path / args.retrieval_path
    questions_name = file_questions_path.split("/")[-1].replace(".json","")
    retrieval_questions_dir = retrieval_dir / questions_name

    mapping_dir = dataset_path / "mapping"
    mapping_file = mapping_dir / "file_mapping.json"

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    texts_to_files = mapping["texts_to_files"]

    with open(file_questions_path, "r", encoding="utf-8") as json_file:
        questions = json.load(json_file)

    retieved_dict = dict()

    for emb in os.listdir(retrieval_questions_dir):
        try:
            retrieved = pickle.load(open(os.path.join(retrieval_questions_dir,emb ),"rb"))
            retieved_dict[emb] = retrieved
        except:
            pass


    N = len(questions)
    accuracies = {}
    retrieval_scores = {}



    for model in retieved_dict:
        accuracies[model] = {}
        retrieved_doc = pickle.load(open(os.path.join(retrieval_questions_dir, model),"rb"))
        dict_answers = []
        for q in questions:
            retrieved_docs = []
            for el in list(retrieved_doc[q['question']]['id'])[:200]:
                id_doc = os.path.basename(texts_to_files[el.split("#")[0]+".txt"])
                if id_doc not in retrieved_docs:
                    retrieved_docs.append(id_doc)
            if type(q['id']) == str:
                actual = [q['id']]
            else:
                actual = q['id']
            dict_answers.append({"predicted": retrieved_docs, 'actual': actual})

        retrieval_scores[model] = retrieval.compute_metrics_for_queries(dict_answers)


    with open(os.path.join(retrieval_questions_dir, "retrieval_scores.json"), "w") as file:
        json.dump(retrieval_scores, file, ensure_ascii=True, indent=4)



    k_values = list(range(1, 11))
    plt.figure(figsize=(12, 7))
    for model, performance in retrieval_scores.items():
        plt.plot(k_values, [performance["recall_at_k"][k] for k in k_values], marker='o', label=model)
    plt.title("Avg recall at k", fontsize=16)
    plt.xlabel("k", fontsize=14)
    plt.ylabel("Avg recall", fontsize=14)
    plt.xticks(k_values)
    plt.legend(title="Models", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig(retrieval_questions_dir / "retrieval_at_k_scores.png", format="png")
    plt.show()
