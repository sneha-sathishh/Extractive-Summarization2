import streamlit as st
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import pymupdf

# Download stopwords if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
from url_summarizer import url_summarizer
from extractive_summarizer import summarizer
from streamlit_option_menu import option_menu

st.set_page_config(layout='wide')
 
st.title('Extractive Summarizer')

if "uploaded_file_sum" not in st.session_state:
    st.session_state["uploaded_file_sum"] = None

if "weburl" not in st.session_state:
    st.session_state["weburl"] = None

col1, col2 = st.columns([1,2])

with st.sidebar:
    selected = option_menu(
        menu_title="Select an Option",
        options=['PDF Summarizer', 'URL Summarizer']
    )
 
if selected == "PDF Summarizer":
    with st.form(key='my_form'):
        st.session_state["uploaded_file_sum"] = st.file_uploader("Upload your file here")
        keywords = st.text_input(label = "Enter the keywords to summarize ", type = "default" )
        lines = st.text_input(label = "Enter the the number of lines ", type = "default" )  
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(summarizer(st.session_state["uploaded_file_sum"], keywords,int(lines)))

elif selected == "URL Summarizer":
        with st.form(key='my_form'):
            st.session_state["weburl"] = st.text_input("weburl", type = "default")
            keywords = st.text_input(label = "Enter the keywords to summarize ", type = "default" )
            lines = st.text_input(label = "Enter the the number of lines ", type = "default" )  
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write(url_summarizer(st.session_state["weburl"], keywords,int(lines)))
            # print(keywords,lines)



