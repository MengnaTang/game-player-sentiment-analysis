# utils.py
# Common functions for embedding generation, cosine similarity, score aggregation, and weighted mean.

import os
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------------------------------------
# Embedding generation
# ----------------------------------------------------------------------

def generate_datasetembeddings(dataset, client):
    """
    Generate OpenAI embeddings for a list of texts.
    dataset: list of strings
    client: OpenAI client
    Returns: list of embedding vectors
    """
    model = "text-embedding-3-large"
    N = len(dataset)
    batch_size = 50
    embed_matrix = []
    for i in range(0, N, batch_size):
        embed_batch = dataset[i:min(N, i+batch_size)]
        embeddings = client.embeddings.create(input=embed_batch, model=model)
        for j in range(len(embed_batch)):
            embed_matrix.append(embeddings.data[j].embedding)
    return embed_matrix


def get_embeddings(input_filename, output_basepath, input_type, text_col='Reviews'):
    """
    Generate embeddings for a CSV file (reviews or queries) and save as pkl.
    input_filename: path to CSV
    output_basepath: base path for output pkl files (without extension)
    input_type: 'review' or 'query'
    text_col: column name containing text (for 'review')
    """
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_API_BASE"))

    batch = 500
    dataset = pd.read_csv(input_filename)
    page = 0
    while page < dataset.shape[0]:
        if page + batch < dataset.shape[0]:
            tmp_dataset = dataset[page: page+batch]
            page += batch
        else:
            tmp_dataset = dataset[page:]
            page = dataset.shape[0]

        if input_type == 'review':
            texts = tmp_dataset[text_col].tolist()
            embeddings = generate_datasetembeddings(texts, client)
            output_file = f"{output_basepath}_{page}.pkl"
        else:  # query
            embeddings = generate_datasetembeddings(tmp_dataset['query'].tolist(), client)
            output_file = f"{output_basepath}.pkl"

        tmp_dataset['embedding'] = embeddings
        if 'Unnamed: 0' in tmp_dataset.columns:
            tmp_dataset = tmp_dataset.drop(columns=['Unnamed: 0'])
        tmp_dataset = tmp_dataset.dropna(how='any')

        with open(output_file, 'wb') as f:
            pickle.dump(tmp_dataset.to_dict(), f)
        print(f"Saved {len(tmp_dataset)} rows to {output_file}")
    print("Embedding generation complete.")


# ----------------------------------------------------------------------
# Cosine similarity
# ----------------------------------------------------------------------

def cosine_similarity(A, B):
    """
    Compute cosine similarity between two sets of vectors.
    A, B: numpy arrays of shape (n, d) and (m, d)
    Returns: matrix of shape (n, m)
    """
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.inner(A, B)


# ----------------------------------------------------------------------
# Load review embeddings (multiple pkl files)
# ----------------------------------------------------------------------

def load_review_embeddings(basepath):
    """
    Load and merge review embedding pkl files created by get_embeddings.
    basepath: base path used in get_embeddings (e.g., 'processed/bilibili_embeddings')
    Returns: DataFrame with all rows and an 'embedding' column.
    """
    dirname = os.path.dirname(basepath)
    basename = os.path.basename(basepath)
    files = [f for f in os.listdir(dirname) if f.startswith(basename) and f.endswith('.pkl')]
    dfs = []
    for f in files:
        with open(os.path.join(dirname, f), 'rb') as fp:
            dfs.append(pd.DataFrame(pickle.load(fp)))
    return pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------------------------
# Subscale score aggregation
# ----------------------------------------------------------------------

def aggregate_scores(df, subscale_mapping):
    """
    Aggregate cosine similarity columns into subscale scores.
    df: DataFrame with cos_sim0...cos_simN columns.
    subscale_mapping: dict {subscale_name: list of column indices}
    Returns: DataFrame with one column per subscale.
    """
    scores = pd.DataFrame()
    for name, idxs in subscale_mapping.items():
        cols = [f'cos_sim{i}' for i in idxs]
        if len(cols) == 1:
            scores[name] = df[cols[0]]
        else:
            scores[name] = df[cols].mean(axis=1)
    return scores


# ----------------------------------------------------------------------
# Weighted mean
# ----------------------------------------------------------------------

def weighted_mean(scores, weights):
    """
    Compute weighted average of scores using weights (e.g., likes).
    scores: array-like
    weights: array-like of same length
    """
    if weights.sum() == 0:
        return scores.mean()
    return np.average(scores, weights=weights)