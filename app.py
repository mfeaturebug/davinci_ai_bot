from helper import *
import streamlit as st

st.title("Ask Davinci!")
# sim_val = st.slider('Select similarity',
#                    min_value=0.0, max_value=1.0, step=0.01)

def removeNonAscii(s): return "".join(i for i in s if ord(i)<126 and ord(i)>31)

def main():
    # Get user input
    user_query = st.sidebar.text_input("Ask the Davinci Bot", value='What is a good product for gut health?')

    if user_query != ":q" or user_query != "":
        # st.write(response['gpt_full_response']['choices'][0]['text'])
        # response = get_contact_response(user_query)
        top_matches = get_top_matches(user_query, n=20)
        # display = response[['similarity', 'product_description', 'product_name', 'product_url']]
        nlp_response = get_gpt_response(query=user_query, emb_df=top_matches.head(3))
        nlp_response = '"' + nlp_response + '"'
        st.write(nlp_response)
        # st.write(top_matches[['product_name', 'product_price',
        #                       'product_url']].head(3))
        return


main()
