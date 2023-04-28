from helper import *
import streamlit as st

# sim_val = st.slider('Select similarity',
#                    min_value=0.0, max_value=1.0, step=0.01)

def removeNonAscii(s): return "".join(i for i in s if ord(i)<126 and ord(i)>31)

st.sidebar.header('Ask Davinci!')
def main():
    # Get user input
    user_query = st.sidebar.text_input("Ask a question about Davinci products", value='What is a good product for gut health?')

    if user_query != ":q" or user_query != "":
        # st.write(response['gpt_full_response']['choices'][0]['text'])
        # response = get_contact_response(user_query)
        with st.spinner('Wait for it...'):
            top_matches = get_top_matches(user_query, n=20)
            nlp_response = get_gpt_response(query=user_query, emb_df=top_matches.head(3))
            nlp_response = '"' + nlp_response + '"'
            st.write(nlp_response)
            # CSS to inject contained in a string
            hide_table_row_index = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        </style>
                        """

            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.subheader('Relevant Product Links')
            # Display a static table
            st.table(top_matches[['product_name', 'product_url']].head(3))
        return


main()
