import time
import google.generativeai as genai
from groq import Groq, APIError
from .key_loader import load_api_keys
from openai import AzureOpenAI, OpenAI
import json

api_keys = load_api_keys()

def get_gemini_response(prompt, model_name, paths=[], initial_wait=1):
    parts = []
    genai.configure(api_key=api_keys["gemini"]["api_key"])

    if isinstance(paths, list) and paths:
        files = []
        for index, path in enumerate(paths):
            print(f"Processing {index + 1} / {len(paths)}")
            attempts, max_attempts, success = 0, 5, False

            while attempts < max_attempts and not success:
                attempts += 1
                try:
                    file = genai.upload_file(path=path)
                    while file.state.name == "PROCESSING":
                        print("Waiting for video processing.")
                        time.sleep(20)
                        file = genai.get_file(file.name)
                    
                    if file.state.name == "FAILED":
                        print(f"Attempt {attempts} failed for {path}.")
                        if attempts == max_attempts:
                            raise ValueError(f"File processing failed after {max_attempts} attempts.")
                    else:
                        success = True
                        files.append(file)
                        print(f"File {path} uploaded successfully on attempt {attempts}.")
                except Exception as e:
                    print(f"Error on attempt {attempts} for {path}: {e}")
                    if attempts == max_attempts:
                        raise ValueError(f"Failed to upload {path} after {max_attempts} attempts.")
        parts += files

    system_prompt = "You are an AI assistant that answers questions following the given instructions."
    parts.append(f"{system_prompt}\n---\n{prompt}")

    wait_time = initial_wait
    model = genai.GenerativeModel(model_name=model_name)

    while True:
        try:
            response = model.generate_content(
                parts,
                generation_config={"temperature": 0, "top_p": 0.1, "max_output_tokens": 1}
            )
            return response.text
        except ResourceExhausted:
            time.sleep(wait_time)
            wait_time *= 2
        except Exception as e:
            raise e

def get_groq_response(prompt, model_name, max_retries=5):
    client = Groq(api_key=api_keys["groq"]["api_key"])
    attempts = 0
    while attempts < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that answers questions following the given instructions."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=1,
                top_p=0.1
            )
            return response.choices[0].message.content
        except APIError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 1))
                print(f"Rate limit exceeded. Retrying in {retry_after} seconds.")
                time.sleep(retry_after)
                attempts += 1
            else:
                print("PROMPT\n",prompt)
                print("PROMPT")
                print(f"Error: {e}")
                time.sleep(60)
                attempts += 1
    raise Exception("Maximum retry attempts reached.")

def get_openai_response(prompt, model_name):
    client = AzureOpenAI(api_key=api_keys["openai"]["api_key"],
                         api_version=api_keys["openai"].get("api_version"),
                         azure_endpoint=api_keys["openai"].get("azure_endpoint")) \
        if "azure_endpoint" in api_keys["openai"] else OpenAI(api_key=api_keys["openai"]["api_key"])

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers questions following the given instructions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_completion_tokens=1,
        top_p=1
    )
    return response.choices[0].message



def answer_question(provider, model_name, prompt):
    if provider == "openai":
        res = get_openai_response(prompt,model_name)
    elif provider == "groq":
        res = get_groq_response(prompt, model_name)
    elif provider == "gemini":
        res = get_gemini_response(prompt, model_name)

    try:
        resp = int(res)
    except:
        resp = 5
    return resp





def get_gemini_question(prompt, model_name, paths=[], initial_wait=1):
    parts = []
    genai.configure(api_key=api_keys["gemini"]["api_key"])

    if isinstance(paths, list) and paths:
        files = []
        for index, path in enumerate(paths):
            print(f"Processing {index + 1} / {len(paths)}")
            attempts, max_attempts, success = 0, 5, False

            while attempts < max_attempts and not success:
                attempts += 1
                try:
                    file = genai.upload_file(path=path)
                    while file.state.name == "PROCESSING":
                        print("Waiting for video processing.")
                        time.sleep(20)
                        file = genai.get_file(file.name)
                    if file.state.name == "FAILED":
                        print(f"Attempt {attempts} failed for {path}.")
                        if attempts == max_attempts:
                            raise ValueError(f"File processing failed after {max_attempts} attempts.")
                    else:
                        success = True
                        files.append(file)
                        print(f"File {path} uploaded successfully on attempt {attempts}.")
                except Exception as e:
                    print(f"Error on attempt {attempts} for {path}: {e}")
                    if attempts == max_attempts:
                        raise ValueError(f"Failed to upload {path} after {max_attempts} attempts.")
        parts += files

    system_prompt = "You are an AI assistant that creates multiple-choice questions based on a given text."
    parts.append(f"{system_prompt}\n---\n{prompt}")

    wait_time = initial_wait
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={"temperature": 0}
    )

    while True:
        try:
            response = model.generate_content(parts)
            return response.text
        except Exception as e:
            time.sleep(wait_time)
            wait_time *= 2

def get_groq_question(prompt, model_name, max_retries=5):
    client = Groq(api_key=api_keys["groq"]["api_key"])
    attempts = 0
    while attempts < max_retries:
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI assistant that creates multiple-choice questions based on a given text."},
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                temperature=0
            )
            return response.choices[0].message.content
        except APIError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 1))
                print(f"Rate limit exceeded. Retrying in {retry_after} seconds.")
                time.sleep(retry_after)
                attempts += 1
            else:
                print(f"Error: {e}")
                break
    raise Exception("Maximum retry attempts reached.")


def get_openai_question(prompt, model_name):
    client = AzureOpenAI(api_key=api_keys["openai"]["api_key"],
                         api_version=api_keys["openai"].get("api_version"),
                         azure_endpoint=api_keys["openai"].get("azure_endpoint")) \
        if "azure_endpoint" in api_keys["openai"] else OpenAI(api_key=api_keys["openai"]["api_key"])

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI assistant that creates multiple-choice questions based on a given text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content



def generate_question(provider, model_name, prompt):
    if provider == "openai":
        res = get_openai_question(prompt,model_name)
    elif provider == "groq":
        res = get_groq_question(prompt, model_name)
    elif provider == "gemini":
        res = get_gemini_question(prompt, model_name)
    res = "{"+"{".join(res.split("{")[1:])
    res = "}".join(res.split("}")[:-1])+"}"
    res = res.replace('"is_correct": true','"is_correct": "True"').replace('"is_correct": false','"is_correct": "False"')
    try:
      resp = json.loads(res)
    except:
      resp = eval(res)
      for option in resp["options"]:
        option["is_correct"] = str(option["is_correct"])
    return resp




