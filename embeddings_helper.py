import streamlit as st
import pandas as pd
import numpy as np
import openai


@st.cache_data
def get_embeddings_data_frame(path):
    df = pd.read_csv(path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    return df


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_ada_embedding(text):
    try:
        return openai.Embedding.create(
            model='text-embedding-ada-002',
            input=text,
        )["data"][0]["embedding"]
    except Exception:
        return False


def get_similarities():
    pass
