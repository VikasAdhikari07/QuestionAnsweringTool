import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_data(urls, file_path):
        
    # load data 
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        
        # split data 
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n','.',','],
            chunk_size = 1000,
            chunk_overlap=100,
        )
        docs = text_splitter.split_documents(data)
        # creating embeddings 

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FAISS.from_documents(docs,embeddings)
        # saving vectors 
        with open(file_path,'wb') as f:
            pickle.dump(vector_store,f)