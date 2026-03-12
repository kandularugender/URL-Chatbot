import os
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

os.environ["USER_AGENT"] = "Mozilla/5.0"

st.title("🌐 Chat with Website")

# URL input
url = st.text_input("Enter Website URL")

# Load website only when button clicked
if st.button("Load Website"):

    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = Chroma.from_documents(
        splits,
        embedding=embeddings
    )

    st.session_state.retriever = vectorstore.as_retriever()

    st.success("Website loaded successfully!")

# Ask question only after website loaded
if "retriever" in st.session_state:

    question = st.text_input("Ask a question about the website")

    if question:

        docs = st.session_state.retriever.invoke(question)

        context = "\n".join([doc.page_content for doc in docs])

        llm = Ollama(model="mistral")

        prompt = f"""
        Answer based only on the context.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.write("### Answer:")
        st.write(response)