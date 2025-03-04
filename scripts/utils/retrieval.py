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

from typing import List, Dict
import math


#import sentence transformers libraries
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from sklearn.feature_extraction.text import TfidfVectorizer

def read_embeddings(folder, special = None):     #special is a list of embedding files
    
    api_keys = load_api_keys()
    embeddings_dict = dict()
    if special == None:
        list_file = os.listdir(folder)
    else:
        list_file = special
    for file in list_file:
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            name = file.replace("|","/")
            name_split = name.split("-")
            type_emb=name_split[0]
            embeddings = pickle.load(open(path,"rb"))
            
            if type_emb == "colbert":
                
                model_name = "-".join(name_split[1:])
                
                embeddings["D"] = torch.from_numpy(embeddings["D"])
                embeddings["D_mask"] = torch.from_numpy(embeddings["D_mask"])

                checkpoint = model_name
                config = ColBERTConfig(doc_maxlen=500, nbits=2)
                ckpt = Checkpoint(checkpoint, colbert_config=config)
                embeddings["ckpt"] = ckpt
            elif type_emb == "sentencetransformers":
                model_name = "-".join(name_split[1:])
                model = SentenceTransformer(model_name)
                embeddings["model"] = model
            elif type_emb == "openai" or type_emb == "openai_512":


                if "azure_endpoint" in api_keys["openai"]:

                    client_emb = AzureOpenAI(
                          api_key = api_keys["openai"]["api_key"],  
                          api_version = api_keys["openai"]["api_version"],
                          azure_endpoint = api_keys["openai"]["azure_endpoint"]
                        )
                else:
                    openai.api_key =  api_keys["openai"]["api_key"]
                    client_emb = OpenAI()

                embeddings["client"] = client_emb
                
            embeddings_dict[name] = embeddings
        
    return embeddings_dict


def retrieve_passages(query, collection_embeddings, k=None):
    
    if "D" in collection_embeddings:
        ckpt = collection_embeddings["ckpt"]
        Q = ckpt.queryFromText([query])
        D = collection_embeddings["D"]
        D_mask = collection_embeddings["D_mask"]
        scores = colbert_score(Q, D, D_mask).flatten().cpu().numpy().tolist()
        ranking = np.argsort(scores)[::-1]
        if k is not None:
            ranking = ranking[:k]
        list_retrieval = [{"id":collection_embeddings["ids"][i], "threshold":scores[i]} for i in ranking]

    elif "TfidfVectorizer" in collection_embeddings:
        X = collection_embeddings["TfidfVectorizer"]
        Q = X.transform([query]).toarray()[0]
        scores = cosine_similarity([Q], collection_embeddings["embeddings"])
        ranking = np.argsort(scores)[0][::-1]
        if k is not None:
            ranking = ranking[:k]
        list_retrieval = [{"id":collection_embeddings["ids"][i], "threshold":scores[0][i]} for i in ranking]
        
    elif "embeddings" in collection_embeddings:
        if "model" in collection_embeddings:
            Q = collection_embeddings["model"].encode(query)
        elif "client" in collection_embeddings:
            response = collection_embeddings["client"].embeddings.create(
                input=query,
                model="embedding"
            )
            Q = response.data[0].embedding  
        scores = cosine_similarity([Q], collection_embeddings["embeddings"])
        ranking = np.argsort(scores)[0][::-1]
        if k is not None:
            ranking = ranking[:k]
        list_retrieval = [{"id":collection_embeddings["ids"][i], "threshold":scores[0][i]} for i in ranking]

    df = pd.DataFrame(list_retrieval)
    df_unique = df.loc[df.groupby('id')['threshold'].idxmax()]
    df_unique = df_unique.sort_values(by='threshold', ascending=False)
    df_unique.reset_index(drop=True, inplace=True)
    return df_unique




#metrics
from typing import List
import math

def precision_at_k(k: int, predicted: List[int], actual: List[int]) -> float:
    """
    Function to compute precision@k given predicted and actual lists

    Inputs:
        k         -> integer number of items to consider, or None to consider all
        predicted -> list of predicted items (ordered by relevance)
        actual    -> list of actual relevant items

    Output:
        Floating-point precision value for k items
    """
    if k is None:
        k = len(predicted)
    elif k <= 0:
        raise ValueError(f"Value of k should be greater than 0, received: {k}")
    
    # Take the top-k items from predicted
    top_k_predicted = predicted[:k]

    # Compute the intersection of top-k predicted items with actual items
    relevant_at_k = set(top_k_predicted) & set(actual)

    # Precision@k is the ratio of relevant items in the top-k predictions
    precision = len(relevant_at_k) / len(top_k_predicted) if len(top_k_predicted) > 0 else 0.0

    return precision



def recall_at_k(k: int, predicted: List[int], actual: List[int]) -> float:
    """
    Function to compute recall@k given predicted and actual lists

    Inputs:
        k         -> integer number of items to consider, or None to consider all
        predicted -> list of predicted items (ordered by relevance)
        actual    -> list of actual relevant items

    Output:
        Floating-point recall value for k items
    """
    if k is None:
        k = len(predicted)
    elif k <= 0:
        raise ValueError(f"Value of k should be greater than 0, received: {k}")
    
    # Take the top-k items from predicted
    top_k_predicted = predicted[:k]

    # Compute the intersection of top-k predicted items with actual items
    relevant_at_k = set(top_k_predicted) & set(actual)

    # Recall@k is the ratio of relevant items found in top-k predictions to the total relevant items
    recall = len(relevant_at_k) / len(actual) if len(actual) > 0 else 0.0

    return recall

def mean_average_precision(predicted: List[int], actual: List[int]) -> float:
    """
    Function to compute Mean Average Precision (MAP) given predicted and actual lists

    Inputs:
        predicted -> list of predicted items (ordered by relevance)
        actual    -> list of actual relevant items

    Output:
        Floating-point MAP value
    """
    average_precision = 0.0
    relevant_items = 0

    for i, pred in enumerate(predicted, start=1):
        if pred in actual:
            relevant_items += 1
            average_precision += relevant_items / i

    return average_precision / len(actual) if actual else 0.0

def mean_reciprocal_rank(predicted: List[int], actual: List[int]) -> float:
    """
    Function to compute Mean Reciprocal Rank (MRR) given predicted and actual lists

    Inputs:
        predicted -> list of predicted items (ordered by relevance)
        actual    -> list of actual relevant items

    Output:
        Floating-point MRR value
    """
    for i, pred in enumerate(predicted, start=1):
        if pred in actual:
            return 1 / i

    return 0.0

def normalized_discounted_cumulative_gain(k: int, predicted: List[int], actual: List[int]) -> float:
    """
    Function to compute Normalized Discounted Cumulative Gain (NDCG) at k

    Inputs:
        k         -> integer number of items to consider, or None to consider all
        predicted -> list of predicted items (ordered by relevance)
        actual    -> list of actual relevant items

    Output:
        Floating-point NDCG value
    """
    if k is None:
        k = len(predicted)

    def dcg(items: List[int], actual: List[int]) -> float:
        return sum(1 / math.log2(idx + 2) for idx, item in enumerate(items) if item in actual)

    top_k_predicted = predicted[:k]
    ideal_order = sorted(actual, key=lambda x: predicted.index(x) if x in predicted else float('inf'))[:k]

    return dcg(top_k_predicted, actual) / dcg(ideal_order, actual) if actual else 0.0


def compute_metrics_for_queries(queries: List[Dict[str, List[int]]]):
    """
    Compute metrics (precision, recall, NDCG for k=1 to 10, MAP, and MRR) across multiple queries.

    Inputs:
        queries -> List of dictionaries, where each dictionary has:
                   'predicted': list of predicted items (ordered by relevance)
                   'actual': list of actual relevant items

    Output:
        Aggregated metrics as a dictionary.
    """
    aggregated_metrics = {
        "precision_at_k": {k: 0.0 for k in range(1, 11)},
        "recall_at_k": {k: 0.0 for k in range(1, 11)},
        "ndcg_at_k": {k: 0.0 for k in range(1, 11)},
        "map": 0.0,
        "mrr": 0.0
    }

    num_queries = len(queries)

    for query in queries:
        predicted = query["predicted"]
        actual = query["actual"]

        # Compute metrics for k=1 to 10
        for k in range(1, 11):
            aggregated_metrics["precision_at_k"][k] += precision_at_k(k, predicted, actual)
            aggregated_metrics["recall_at_k"][k] += recall_at_k(k, predicted, actual)
            aggregated_metrics["ndcg_at_k"][k] += normalized_discounted_cumulative_gain(k, predicted, actual)

        # Compute MAP and MRR for this query
        aggregated_metrics["map"] += mean_average_precision(predicted, actual)
        aggregated_metrics["mrr"] += mean_reciprocal_rank(predicted, actual)

    # Average metrics over the number of queries
    for k in range(1, 11):
        aggregated_metrics["precision_at_k"][k] /= num_queries
        aggregated_metrics["recall_at_k"][k] /= num_queries
        aggregated_metrics["ndcg_at_k"][k] /= num_queries

    aggregated_metrics["map"] /= num_queries
    aggregated_metrics["mrr"] /= num_queries


#    p,r,f=search_max_th(queries, -100, 100)

#    aggregated_metrics["precision_at_th"] = p
#    aggregated_metrics["recall_at_th"] = r
#    aggregated_metrics["f1_at_th"] = f


    return aggregated_metrics



    