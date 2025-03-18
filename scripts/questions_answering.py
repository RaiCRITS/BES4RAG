from utils import llms
from utils import prompt
import argparse
import json
import pickle
from tqdm import tqdm
from pathlib import Path
import os

def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def main():
    """Main function to process files in the dataset directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--questions_path", type=str, default="questions", help="Folder for converted questions")
    parser.add_argument("--name_file_questions", type=str, default="generated_questions.json", help="Name of JSON file containing questions")
    parser.add_argument("--retrieval_path", type=str, default="retrieval", help="Folder for retrieval results")
    parser.add_argument("--llm_provider", type=str, default="groq", help="LLM provider (e.g openai, gemini, groq)")
    parser.add_argument("--model_name", type=str, default="llama3-70b-8192", help="The name of the specific model")
    parser.add_argument("--answers_path", type=str, default="answers", help="Folder for answers")
    parser.add_argument("--skip_existing", type=bool, default=True, nargs="?", const=True, help="Skip processing if passages.json already exists")


    args = parser.parse_args()

    provider = args.llm_provider
    model_name = args.model_name
    
    dataset_path = Path(args.dataset_path)
    questions_dir = dataset_path / args.questions_path
    file_questions_path = questions_dir / args.name_file_questions
    retrieval_dir = dataset_path / args.retrieval_path
    questions_name = str(file_questions_path).split("/")[-1].replace(".json","")
    retrieval_questions_dir = retrieval_dir / questions_name
    answers_dir = dataset_path / args.answers_path

    ensure_dir(answers_dir)

    answers_questions_dir = answers_dir / questions_name

    ensure_dir(answers_questions_dir)

    llm_provider = args.llm_provider
    model_name = args.model_name

    with open(file_questions_path, "r", encoding="utf-8") as json_file:
        questions = json.load(json_file)


    retieved_dict = dict()

    for emb in os.listdir(retrieval_questions_dir):
        try:
            retrieved = pickle.load(open(os.path.join(retrieval_questions_dir,emb ),"rb"))
            retieved_dict[emb] = retrieved
        except:
            pass


    ks = [x for x in range(6)]+ [10]


    for emb in retieved_dict:
        dict_answers = {}

        file_path = os.path.join(answers_questions_dir, emb + ".json")

        if os.path.exists(file_path) and args.skip_existing:
            dict_answers = json.load(open(file_path))
            
        retrieved = retieved_dict[emb]
        for q in tqdm(questions):
            if q['question'] not in dict_answers:
                dict_answers[q['question']] = {}
                dict_answers[q['question']]['correct_answer'] = next((index for index, item in enumerate(q['options']) if item['is_correct'] == 'True'), None)
                resps = []
                for k in ks:    #k=0 is no rag
                    if k == 0:
                        prompt_ = prompt.prompt_question_passages(q, retrieved, k, args.dataset_path, emb, rag = False)
                    else:
                        prompt_ = prompt.prompt_question_passages(q, retrieved, k, args.dataset_path, emb, rag = True)
                    resp = llms.answer_question(provider,model_name,prompt_)
                    resps.append({"k":k, "answer":resp})
                dict_answers[q['question']]['answer_LLM'] = resps
                with open(file_path, 'w') as json_file:
                    json.dump(dict_answers, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()

