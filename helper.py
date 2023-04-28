import numpy as np
import openai
import pandas as pd
import streamlit as st
from embeddings_helper import *
from ast import literal_eval

# import os
# import glob
#
# csv_dir = os.path.join("./embeddings/", "*.csv")
# files = glob.glob(csv_dir)
# print(files)
# # joining files with concat and read_csv
# df = pd.concat(map(pd.read_csv, files), ignore_index=True)
embeddings_path = "./embeddings/davinci_embeddings_v2_0_262.csv"

df = get_embeddings_data_frame(embeddings_path)
df['product_details_string'] = df['product_details'].apply(lambda x: ''.join(literal_eval(x)))


def get_query_classification(query):
    response = {}
    prompt_prefix = '"""You will help classify the text into one of six topics of Reviews, Ingredients, Products, Pricing, Use Case or Other. Use the following examples:\nWhich products have Vitamin C?: ' \
                    'Ingredients\nWhat is the best product for skin health?: Use Case\nWhat is Collagen?: Use ' \
                    'Case\nHow much does the collagen product cost?: Pricing\nWhich collagen product has the maximum ' \
                    'reviews?: Reviews\nWhere can I buy your products?: Other"""\nText:\n'
    prompt = prompt_prefix + query + '"""\n'
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            # engine='text-curie-001',
            prompt=prompt,
            temperature=0.4,
            max_tokens=200,
        )
    except Exception:
        response['choices'][0]['text'] = 'Something seems to have gone wrong. I am at my wits end!!'
    return response['choices'][0]['text']


def analyze_query_by_classification(query, classification='Other', n=3):
    response = {}
    try:
        classification = get_query_classification(query)
    except Exception:
        classification = 'Other'
    query_embeddings = get_ada_embedding(query)
    try:
        if classification == 'Use Case':
            df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embeddings))
            results = (
                df.sort_values("similarity", ascending=False)
                .head(n)
            )
        elif classification == 'Ingredients':
            df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embeddings))
            results = (
                df.sort_values("similarity", ascending=False)
                .head(n)
            )
        elif classification == 'Product':
            df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embeddings))
            results = (
                df.sort_values("similarity", ascending=False)
                .head(n)
            )
        elif classification == 'Reviews':
            df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embeddings))
            results = (
                df.sort_values("similarity", ascending=False)
                .head(n)
            )
        elif classification == 'Pricing':
            df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embeddings))
            results = (
                df.sort_values("similarity", ascending=False)
                .head(n)
            )
        elif classification == 'Other':
            df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embeddings))
            results = (
                df.sort_values("similarity", ascending=False)
                .head(n)
            )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        results = False
    return results


def get_gpt_response(query, emb_df):
    response = {}
    emb_df.reset_index(inplace=True)
    prompt_prefix = '"""Based on the multiple product information below, respond to the users query as precisely as ' \
                    'possible. Include product name, price and url in the response when available. Also include alternate products if available and separate the response in paragraphs. Otherwise say: Hmm, I am unable to respond to that.'
    prompt_suffix = '\nQuery: ' + query + "\n"
    prompt = ''
    for index, row in emb_df.iterrows():
        text = '\n\n' + str(index + 1) + ')' + row["product_name"] + ':\nDescription: ' + row["product_description"] + \
               '\nProduct Details: ' + row['product_details_string'] + '\nPrice: ' + row[
                   'product_price'] + '\Product url: ' + row['product_url']

        prompt = prompt + text

    prompt = prompt_prefix + prompt + '"""\n' + prompt_suffix
    try:
        response['gpt_full_response'] = openai.Completion.create(
            engine="text-davinci-003",
            # engine='text-curie-001',
            prompt=prompt,
            temperature=0.7,
            max_tokens=1200,
        )
    except Exception:
        response['gpt_full_response']['choices'][0][
            'text'] = 'Something seems to have gone wrong. I am at my wits end!!'
    print(response['gpt_full_response']['choices'][0]['text'])
    return response['gpt_full_response']['choices'][0]['text']


def get_top_matches(query, n=10):
    query_embeddings = get_ada_embedding(query)
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embeddings))
    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
    )
    return results
