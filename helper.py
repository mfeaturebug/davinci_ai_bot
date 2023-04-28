import numpy as np
import openai
import pandas as pd
import streamlit as st
import time


# import os
# import glob
#
# csv_dir = os.path.join("./embeddings/", "*.csv")
# files = glob.glob(csv_dir)
# print(files)
# # joining files with concat and read_csv
# df = pd.concat(map(pd.read_csv, files), ignore_index=True)

@st.cache_data
def get_embeddings_data_frame():
    print("Method: get_embeddings_data_frame")
    embeddings_path = "./embeddings/davinci_embeddings_0_262.csv"
    df = pd.read_csv(embeddings_path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    return df


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# search through the reviews for a specific product
def get_semantic_records(df, query, n=3):
    print("Method: get_semantic_records")
    try:
        query_embedding = openai.Embedding.create(
            model='text-embedding-ada-002',
            input=query,
        )["data"][0]["embedding"]
        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embedding))
        results = (
            df.sort_values("similarity", ascending=False)
            .head(n)
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        results = 'Something went wrong'
    return results


def get_gpt_response(context, query):
    print("Method: get_gpt_response")
    context = context.iloc[0]
    prefix = 'Use the below text to answer the subsequent question in under 10 sentences. If the answer cannot be found in the text, write "I could not find an answer."'
    prefix = prefix + '\n\nText: ' + context['text']
    question = "\n\n Question: " + query
    prompt = prefix + question
    response = {
        'chunk': context['file_name'],
        'similarity': context['similarity'],
        'file_path': context['file_path']
    }
    # print(prompt)
    try:
        response['gpt_full_response'] = openai.Completion.create(
            engine="text-davinci-003",
            # engine='text-curie-001',
            prompt=prompt,
            temperature=0.6,
            max_tokens=200,
        )
    except Exception:
        response['gpt_full_response'] = 'Something seems to have gone wrong. I am at my wits end!!'
    return response


def get_contact_response(query):
    print("Method: get_contact_response")
    df = get_embeddings_data_frame()
    records = get_semantic_records(df, query, n=50)

    return records
