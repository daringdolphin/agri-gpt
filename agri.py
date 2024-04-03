__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import re
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from utils import clean_text, load_documents, embed_document
if 'embed_model' not in st.session_state:
    st.session_state.embed_model = 0

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Streamlit app title and document overview
st.title("🌳Agri-GPT🌳")

docs = load_documents()
docs[0].page_content = clean_text(docs[0].page_content)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# User input for query

db = embed_document(docs)
retriever = db.as_retriever()

st.markdown("""
---
### About
Your AI-powered guide to navigating agricultural emissions reporting, simplifying GHG Protocol guidance into actionable insights.""")

user_query = st.text_input("Enter your query here:", "What are major sources of agricultural emissions and how do I account for these emissions?")

# Search button
if user_query:
    with st.spinner("Searching the document... Please wait."):
        template = """Provide answers that are formatted simply for ease of understanding. Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model='gpt-4-turbo-preview')

        retrieval_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        response = retrieval_chain.invoke(user_query)
        # Display results if any
        if response:
            st.write(response)

# Optional: About section or additional instructions

st.markdown("""---""")

with st.expander("Selected relevant context in guidance document"):
    results = retriever.invoke(user_query)
    for chunks in results:
        st.write(chunks.page_content)

st.markdown("""---""")


