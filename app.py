from helper import *
import streamlit as st

import streamlit as st
from helper import *

st.title("Ask about your CRM contacts!")
# sim_val = st.slider('Select similarity',
#                    min_value=0.0, max_value=1.0, step=0.01)

def main():
    # Get user input
    user_query = st.text_input("Ask the CRM", value='best product for skin health?')

    if user_query != ":q" or user_query != "":
        # st.write(response['gpt_full_response']['choices'][0]['text'])
        response = get_contact_response(user_query)
        top_matches = response[response.similarity >= 0.75]
        display = top_matches[['similarity', 'product_description', 'product_name', 'product_url']]
        st.write(display)
        return


main()
