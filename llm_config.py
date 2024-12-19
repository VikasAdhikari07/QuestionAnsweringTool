from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

def config_groq(temp):
    llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=temp)
    return llm