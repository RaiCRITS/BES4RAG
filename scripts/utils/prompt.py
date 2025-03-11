import os
import json


def prompt_1_text(text):
    return f"""Create a multiple-choice question  based solely on the content of the following text:
----------------------------
{text}
----------------------------
The question must be written in the same language of the text given and will be used for a test in which the article is not provided, so it must be generic and must not contain references to the article itself (e.g., phrases like "in the given article..." or "based on the text" nor "mentioned in the text" should not appear).

Moreover, if the text mentions a specific event, the question should include as many details as possible. For example, if it mentions a war, the question must clearly specify which war is being referred to. Similarly, any temporal references about the event should be provided—e.g., if the article talks about "today's event," the question must specify "the event on day X," where X is the date of the event, if it can be inferred from the article.

Next, create four possible answer options in the same language of the article, one of which must be correct, while the others should be plausible but incorrect, ensuring the correct answer is unambiguous.

The correct answer must be derivable solely from the content of the article.

For each option, provide an explanation of why, based on the article, the option is either correct or incorrect.

Present the response in the following format:
{{
    "question": <multiple-choice question>,
    "options":[
        {{
            "text":"<option1>",
            "is_correct": "<True/False>",
            "explanation": "<explanation of why option 1 is True or False>"
        }},
        {{
            "text":"<option2>",
            "is_correct": "<True/False>",
            "explanation": "<explanation of why option 2 is True or False>"
        }},
        {{
            "text":"<option3>",
            "is_correct": "<True/False>",
            "explanation": "<explanation of why option 3 is True or False>"
        }},
        {{
            "text":"<option4>",
            "is_correct": "<True/False>",
            "explanation": "<explanation of why option 4 is True or False>"
        }}
    ]
}}
Provide only the question and options formatted as specified above and written in the same language of the text without any additional text."""

def from_id_to_article(id, path):
    with open(os.path.join(path, "texts", id+".txt"), "r") as file:
        text = file.read()
    try:
        with open(os.path.join(path, "descriptions", id+".txt"), "r") as file:
            description = file.read()
        full_text = description + "\n\n" + text
        return full_text
    except:
        return text

def rag_context(question, retrieved,path, k):
    ids_retrieved = retrieved[question]['id'].values[:k] 
    list_retrieval = [from_id_to_article(id, path) for id in ids_retrieved]
    context = ""
    for el in list_retrieval:
        context +=  el + "\n-------------------------\n"
    return context


def prompt_question(q, retrieved, k, path,rag = True):
    question = q['question']
    list_ids = []
    # Crea una stringa dinamica per le opzioni
    multiple_choice = f"{question}:\n"
    
    # Aggiungi ogni opzione dinamicamente, numerandola
    for idx, option in enumerate(q['options']):
        multiple_choice += f"{idx}: {option['text']}\n"
    if rag ==False:
        prompt = f"""Answer the following multiple-choice question:

{multiple_choice}

Respond by providing only the numerical identifier of the correct answer from the options 0, 1, 2, 3. Do not respond with anything other than one of these numbers even if you do not know the answer.
"""  
    else:
        prompt = f"""Answer the following multiple-choice question:

{multiple_choice}

using the following textual documents as possible sources:

*****
{rag_context(question, retrieved,path, k)}*****
  
Respond by providing only the numerical identifier of the correct answer from the options 0, 1, 2, 3. Do not respond with anything other than one of these numbers even if you do not know the answer.
"""
    return prompt



def from_id_to_article_passages(id, path, emb):
    with open(os.path.join(path, "passages", "passages.json"), "r") as file:
        passages = json.load(file)

    p1,p2 = id.split("#")
    text = passages[emb][p1][int(p2)]
    return text

def from_id_to_article_passages_fever(id, passages, emb):
    p1,p2 = id.split("#")
    text = passages[emb][p1][int(p2)]
    return text

def rag_context_passages_fever(question, retrieved,passages, k, emb):
    ids_retrieved = retrieved[question]['id'].values[:k] 
    list_retrieval = [from_id_to_article_passages_fever(id, passages,emb) for id in ids_retrieved]
    context = ""
    for el in list_retrieval:
        context +=  el + "\n-------------------------\n"
    return context
    

def rag_context_passages(question, retrieved,path, k, emb):
    ids_retrieved = retrieved[question]['id'].values[:k] 
    list_retrieval = [from_id_to_article_passages(id, path,emb) for id in ids_retrieved]
    context = ""
    for el in list_retrieval:
        context +=  el + "\n-------------------------\n"
    return context


def prompt_question_passages(q, retrieved, k, path, emb, rag = True):
    question = q['question']
    list_ids = []
    # Crea una stringa dinamica per le opzioni
    multiple_choice = f"{question}:\n"
    
    # Aggiungi ogni opzione dinamicamente, numerandola
    for idx, option in enumerate(q['options']):
        multiple_choice += f"{idx}: {option['text']}\n"
    if rag == False:
        prompt = f"""Answer the following multiple-choice question:

{multiple_choice}

Respond by providing only the numerical identifier of the correct answer from the options 0, 1, 2, 3. Do not respond with anything other than one of these numbers even if you do not know the answer.
"""  
    else:
        prompt = f"""Answer the following multiple-choice question:

{multiple_choice}

using the following textual documents as possible sources:

*****
{rag_context_passages(question, retrieved,path, k, emb)}*****
  
Respond by providing only the numerical identifier of the correct answer from the options 0, 1, 2, 3. Do not respond with anything other than one of these numbers even if you do not know the answer.
"""
    return prompt


def prompt_question_passages_fever(q, retrieved, k, passages, emb, rag = True):
    question = q['question']
    list_ids = []
    # Crea una stringa dinamica per le opzioni
    multiple_choice = f"{question}:\n"
    
    # Aggiungi ogni opzione dinamicamente, numerandola
    for idx, option in enumerate(q['options']):
        multiple_choice += f"{idx}: {option['text']}\n"
    if rag == False:
        prompt = f"""Answer the following multiple-choice question:

{multiple_choice}

Respond by providing only the numerical identifier of the correct answer from the options 0, 1, 2, 3. Do not respond with anything other than one of these numbers even if you do not know the answer.
"""  
    else:
        prompt = f"""Answer the following multiple-choice question:

{multiple_choice}

using the following textual documents as possible sources:

*****
{rag_context_passages_fever(question, retrieved,passages, k, emb)}*****
  
Respond by providing only the numerical identifier of the correct answer from the options 0, 1, 2. Do not respond with anything other than one of these numbers even if you do not know the answer.
"""
    return prompt

def prompt_question_passages_fever_V2(q, retrieved, k, passages, emb, rag = True):
    question = q['question']
    list_ids = []
    # Crea una stringa dinamica per le opzioni
    multiple_choice = f"{question}:\n"
    
    # Aggiungi ogni opzione dinamicamente, numerandola
    for idx, option in enumerate(q['options']):
        multiple_choice += f"{idx}: {option['text']}\n"
    if rag == False:
        prompt = f"""Consider the following claim and establish if it is True, False or if you do not have enough information to answer.

{multiple_choice}

Respond by providing only the numerical identifier of the correct answer from the options 0, 1, 2. Do not respond with anything other than one of these numbers even if you do not know the answer.
"""  
    else:
        prompt = f"""Consider the following claim and establish if it is True, False or if you do not have enough information to answer.

{multiple_choice}

use the following textual documents, if usefull, as sources to establish the truthfullness of the claim.

*****
{rag_context_passages_fever(question, retrieved,passages, k, emb)}*****
  
Respond by providing only the numerical identifier of the correct answer from the options 0, 1, 2. Do not respond with anything other than one of these numbers even if you do not know the answer.
"""
    return prompt




def gen_prompt_sum(doc):
    return f"""Consider the following document.

*****
{doc}
*****

Write a very short summary of it containing the most important informations in the same language that it is written.
Do not repeat any part of the text contained in the given document but create a new text containing all the information contained in the original one.
Return only the brief summary, written in the same language as the original document, without adding further text.
Summary:
"""




