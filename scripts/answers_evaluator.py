import os
import json
import pickle
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def accuracy_answers(data):
    result_k = {}
    n_questions = len(data)
    
    for question, details in data.items():
        correct_answer = details['correct_answer']
        for answ in details['answer_LLM']:
            k = answ['k']
            answer = answ['answer']
            
            if k not in result_k:
                result_k[k] = 0
            
            if answer == correct_answer:
                result_k[k] += 1
    
    # Calcola l'accuracy per ogni valore di k
    accuracy_k = {k: correct / n_questions for k, correct in result_k.items()}
    return accuracy_k


def main():
    """Main function to process files in the dataset directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--questions_path", type=str, default="questions", help="Folder for converted questions")
    parser.add_argument("--name_file_questions", type=str, default="generated_questions.json", help="Name of JSON file containing questions")
    parser.add_argument("--answers_path", type=str, default="answers", help="Folder for answers")


    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    questions_dir = dataset_path / args.questions_path
    file_questions_path = questions_dir / args.name_file_questions
    questions_name = file_questions_path.split("/")[-1].replace(".json","")
    answers_dir = dataset_path / args.answers_path
    answers_questions_dir = answers_dir / questions_name

    with open(file_questions_path, "r", encoding="utf-8") as json_file:
        questions = json.load(json_file)



    answers_dict = dict()

    for emb_f in os.listdir(answers_questions_dir):
        emb = emb_f.replace(".json","")
        if emb != "accuracy":
            try:
                answers = json.load(open(os.path.join(answers_questions_dir,emb+".json" )))
                answers_dict[emb] = retrieved
            except:
                pass


    accuracy_dict = {}
    for emb in answers_dict:
        answers = answers_dict[emb]
        accuracy_dict[emb] = accuracy_answers(answers)

    with open(os.path.join(answers_questions_dir,"accuracy.json"), "w") as file:
        json.dump(accuracy_dict, file, ensure_ascii=True, indent=4)


    plt.figure(figsize=(12, 6))

    x_ticks = ["no rag", 1, 2, 3, 4, 5, 10]

    for model, results in accuracy_dict.items():
        x = list(results.keys()) 
        y = list(results.values()) 
        plt.plot(x, y, marker='o', label=model)

    plt.xticks(ticks=list(list(accuracy_dict.values())[0].keys()), labels=x_ticks)
    plt.xlabel('k documents used in rag', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy of answers', fontsize=14)
    plt.legend(title='Model', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Mostrare il grafico
    plt.tight_layout()
    plt.savefig(os.path.join(answers_questions_dir,"accuracy.png"), format="png")
    plt.show()
