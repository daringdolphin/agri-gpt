import re
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

@st.cache_resource
def embed_document(_docs):
    ''' 
    This function generates embeddings for the documents and returns the vectorstore as a Chroma object
    '''
    db = Chroma.from_documents(_docs, OpenAIEmbeddings())
    st.session_state.embed_model += 1
    return db

def load_documents():
    loader = TextLoader("GHG_protocol_agri_guidance.txt", encoding='utf-8')
    text_documents = loader.load()
    return text_documents

def clean_text(text):
    # Remove formfeed character
    cleaned_text = re.sub(r'\x0c', ' ', text)
    # Remove specific Unicode characters and all outside ASCII range
    cleaned_text = re.sub(r'[\uf0b0-\uf0bf]', ' ', cleaned_text)
    cleaned_text = re.sub(r'[^\x00-\x7F]', ' ', cleaned_text)
    # Remove all line breaks and links
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)
    cleaned_text = re.sub(r'http[s]?://\S+', ' ', cleaned_text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # This line should operate on `cleaned_text`, not `text`
    return cleaned_text


