import os 
import pickle
import time
import langchain
from llm_config import config_groq
from url_to_vector_index import load_data
from langchain.chains import RetrievalQAWithSourcesChain


import streamlit as st


st.title("NEWS Article Question Answering Tool")
st.sidebar.title("NEWS article Links")

llm = config_groq(temp=0.8)

urls=[]

for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector_index.pkl"
main_placeholder = st.empty()

if process_url_clicked:

    load_data(urls=urls, file_path=file_path)
    main_placeholder.text_input("Data is loading...")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            vector_store= pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_store.as_retriever())
            result = chain({"question":query},return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["answer"])