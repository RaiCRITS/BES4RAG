from .key_loader import load_api_keys 

import os
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import time
import random
import shutil
import torch
import pickle
from copy import deepcopy
import numpy as np
import tiktoken

# import colbert libraries
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score

# import openai libraries
from openai import AzureOpenAI
from openai import OpenAI

#import sentence transformers libraries
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer

import re

from nltk.corpus import stopwords

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_tfidf_string(s):
    mapping = {
        "stopwords": "stop_words",
        "maxdf": "max_df",
        "mindf": "mind_df",
        "ngramrange": "ngram_range",
        "maxfeatures": "max_features"
    }

    # Rimuoviamo il prefisso 'TFIDF/' se presente
    s = s.split("/", 1)[-1]

    result = {}
    for key, value in re.findall(r'([^=]+)=([^-=]+)', s):
        key = mapping.get(key.strip("-"), key.strip("-"))  # Mappiamo il nome corretto
        
        if key == "stop_words":
            result[key] = value.split("&")  # Dividiamo le lingue in lista
        elif key == "max_df":
            result[key] = int(value) / 100
        elif key == "mind_df":
            result[key] = int(value) // 100
        elif key == "ngram_range":
            result[key] = tuple(map(int, [value[:2], value[2:]]))
        elif key == "max_features":
            result[key] = int(value)
        else:
            result[key] = value

    return result

def get_tokenizer(embedding_model, TOKENIZER_CACHE):
    """
    Ottiene il tokenizer corretto e il massimo numero di token dal modello di embedding.
    Salva il tokenizer e max_token nella cache globale.
    """

    model_type = embedding_model['type']
    model_name = embedding_model['model_name']

    name_model = "-".join([model_type, model_name.replace("/", "|")])

    if name_model in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[name_model]['tokenizer'], TOKENIZER_CACHE[name_model]['max_token']

    if model_type == "sentencetransformers":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = SentenceTransformer(model_name)
        max_token = model.max_seq_length
    elif model_type == "openai":
        tokenizer = tiktoken.encoding_for_model(model_name)
        max_token = 8191
    elif model_type == "openai512":
        tokenizer = tiktoken.encoding_for_model(model_name)
        max_token = 512
    elif model_type == "colbert":
        config = ColBERTConfig(doc_maxlen=500, nbits=2)
        ckpt = Checkpoint(model_name, colbert_config=config)
        tokenizer = ckpt.doc_tokenizer.tok
        max_token = 500 #config.doc_maxlen  # Assumiamo che il limite massimo sia definito in `doc_maxlen`
    elif model_type == "TFIDF":

        parsed_tfidf = parse_tfidf_string(model_name)
        if "stop_words" in parsed_tfidf:
            stop_words = list()
            for stopwords_lang in parsed_tfidf["stop_words"]:
                stop_words += stopwords.words(stopwords_lang)
            parsed_tfidf["stop_words"] = stop_words
        vectorizer_params = {k: v for k, v in parsed_tfidf.items() if k in ["stop_words", "max_df", "min_df", "ngram_range", "max_features"]}
        tokenizer = TfidfVectorizer(**vectorizer_params)
        max_token = vectorizer_params["max_features"] #todetermine
    else:
        raise ValueError("Model type not supported.")

    
    TOKENIZER_CACHE[name_model] = {'tokenizer': tokenizer, 'max_token': max_token}
    return  tokenizer, max_token


def tokenize(embedding_model, text, TOKENIZER_CACHE):
    """Tokenizza un testo in base al modello specificato."""
    tokenizer, _ = get_tokenizer(embedding_model, TOKENIZER_CACHE)
    model_type = embedding_model['type']
    if model_type == "sentencetransformers":
        return tokenizer(text, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    elif model_type == "openai":
        return tokenizer.encode(text)
    elif model_type == "openai512":
        return tokenizer.encode(text)
    elif model_type == "colbert":
        return tokenizer.tokenize(text)
    elif model_type == "TFIDF":
        return text.split()
    else:
        raise ValueError("Model type not supported.")

def detokenize(embedding_model, tokens, TOKENIZER_CACHE):
    """Ricostruisce il testo dai token in base al modello specificato."""
    tokenizer, _ = get_tokenizer(embedding_model, TOKENIZER_CACHE)
    model_type = embedding_model['type']

    if model_type == "sentencetransformers":
        return tokenizer.decode(tokens)
    elif model_type == "openai":
        return tokenizer.decode(tokens)
    elif model_type == "openai512":
        return tokenizer.decode(tokens)
    elif model_type == "colbert":
        return tokenizer.convert_tokens_to_string(tokens)
    elif model_type == "TFIDF":
        return " ".join(tokens)
    else:
        raise ValueError("Model type not supported.")

def split_tokens(max_tokens, tokens, overlap=40):
    """Divide i token in segmenti con sovrapposizione."""
    segments = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        segments.append(tokens[start:end])
        start += max_tokens - overlap

    return segments

def split_text(embedding_model, text, TOKENIZER_CACHE, max_tokens=None, description = None, overlap=40):
    """
    Divide un testo in segmenti basati sul modello di embedding.
    Se `max_tokens` non è fornito, utilizza il massimo numero di token del modello.
    """
    tokenizer, model_max_token = get_tokenizer(embedding_model, TOKENIZER_CACHE)
    max_tokens = max_tokens or model_max_token

    if description != None:
        max_tokens = max_tokens - len(tokenize(embedding_model, description, TOKENIZER_CACHE))-2    #il meno due è dovuto ai due a capo che incollo nel caso in cui ci sia 
    tokens = tokenize(embedding_model, text, TOKENIZER_CACHE)
    segments_tokens = split_tokens(max_tokens, tokens, overlap=overlap)
    if description != None:
        return [description + "\n\n" + detokenize(embedding_model, segment, TOKENIZER_CACHE) for segment in segments_tokens]
    else:
        return [detokenize(embedding_model, segment, TOKENIZER_CACHE) for segment in segments_tokens]


def compute_embeddings(transcripts,embedding):

    api_keys = load_api_keys()

    name = "-".join([embedding['type'],embedding["model_name"].replace("/","|")])


    if embedding['type'] == "sentencetransformers":
        # Caricare il modello su CPU
        model = SentenceTransformer(embedding['model_name'], device=device)
        
        ids = []
        embeddings = []
        
        for tr in tqdm(transcripts):
            if tr['text'] != "":
                try:
                    ids.append(tr['id'])
                    embeddings.append(model.encode(tr['text'], device=device))  # Forza il calcolo su CPU
                except Exception as e:
                    print(e)
                    print(tr['id'])
        
        return {"embeddings": embeddings, "ids": ids}

        """
        model = SentenceTransformer(embedding['model_name'])
        ids = []
        embeddings = []
        for tr in tqdm(transcripts):
            if tr['text'] != "":
                try:
                    ids.append(tr['id'])
                    embeddings.append(model.encode(tr['text']))    
                except Exception as e:
                    print(e)
                    print(tr['id'])
        return {"embeddings":embeddings,"ids":ids}
        """

    elif embedding['type'] == "openai" or embedding['type'] == "openai512":

        if "azure_endpoint" in api_keys["openai"]:
            client_emb = AzureOpenAI(
                  api_key = api_keys["openai"]["api_key"],  
                  api_version = api_keys["openai"]["api_version"],
                  azure_endpoint = api_keys["openai"]["azure_endpoint"]
                )
        else:
            openai.api_key =  api_keys["openai"]["api_key"]
            client_emb = OpenAI()
        ids = []
        embeddings = []
        for tr in tqdm(transcripts):
            if tr['text'] != "":
                try:
                    ids+= [tr['id']]
                    response = client_emb.embeddings.create(
                            input=tr['text'],
                            model="embedding"
                        )
                    embeddings+= [np.array(response.data[0].embedding)]
                except Exception as e:
                    print(e)
                    print(tr['id'])
        return {"embeddings":embeddings,"ids":ids}


    elif embedding['type'] == "colbert":
        checkpoint = embedding['model_name']
        config = ColBERTConfig(doc_maxlen=500, nbits=2)
        ckpt = Checkpoint(checkpoint, colbert_config=config)
        
        ids = []
        processed_passages = []

        for tr in tqdm(transcripts):
            if tr['text'] != "":
                ids.append(tr['id'])
                processed_passages.append(tr['text'])

        if device == "cpu":
            if hasattr(ckpt, "to"):
                ckpt.to(device)
            with torch.no_grad():
                D = ckpt.docFromText(processed_passages, bsize=32)[0].to(device)
            D_mask = torch.ones(D.shape[:2], dtype=torch.long, device=device)
        else:
            D = ckpt.docFromText(processed_passages, bsize=32)[0]
            D_mask = torch.ones(D.shape[:2], dtype=torch.long)
        D = D.detach().cpu().numpy()
        D_mask = D_mask.detach().cpu().numpy()
        return {"D": D, "D_mask": D_mask, "ids": ids}

    elif embedding['type'] == "TFIDF":
        parsed_tfidf = parse_tfidf_string(embedding['model_name'])
        if "stop_words" in parsed_tfidf:
            stop_words = list()
            for stopwords_lang in parsed_tfidf["stop_words"]:
                stop_words += stopwords.words(stopwords_lang)
            parsed_tfidf["stop_words"] = stop_words
        vectorizer_params = {k: v for k, v in parsed_tfidf.items() if k in ["stop_words", "max_df", "min_df", "ngram_range", "max_features"]}
        X = TfidfVectorizer(**vectorizer_params)
        ids = []
        processed_passages = []
        for tr in tqdm(transcripts):
            if tr['text'] != "":
                    ids.append(tr['id'])
                    processed_passages.append(tr['text'])
        X = X.fit(processed_passages)
        embeddings = list(X.transform(processed_passages).toarray())
        return {"TfidfVectorizer":X,"embeddings":embeddings,"ids":ids}
    else:
        return "model not available"




    """
    elif embedding['type'] == "colbert":
        checkpoint = embedding['model_name']
        config = ColBERTConfig(doc_maxlen=500, nbits=2)
        ckpt = Checkpoint(checkpoint, colbert_config=config)
        ids = []
        D = []
        D_mask = []
        processed_passages = []
        for tr in tqdm(transcripts):
            if tr['text'] != "":
                    ids.append(tr['id'])
                    processed_passages.append(tr['text'])

        D = ckpt.docFromText(processed_passages, bsize=32)[0]
        D_mask = torch.ones(D.shape[:2], dtype=torch.long)
        D = D.detach().cpu().numpy()
        D_mask = D_mask.detach().cpu().numpy()
        
#        D = np.concatenate(D, axis=0)
#        D_mask = np.concatenate(D_mask, axis=0)
        return {"D":D,"D_mask":D_mask,"ids":ids}
    """



