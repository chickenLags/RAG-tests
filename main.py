# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1. Laden van PDF's en splitsen in chunks
def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

# 2. PDF -> tekstchunks -> FAISS index
def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# 3. Vraag beantwoorden
def ask_question(query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print("🔹", doc.page_content[:300])

    # llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    llm = OllamaLLM(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.invoke(query)

if __name__ == "__main__":
    load_dotenv()
    if not os.path.exists("faiss_index"):
        print("📚 Index nog niet gevonden, maak vectorstore aan...")
        docs = load_documents_from_folder("pdfs")
        vectorstore = create_vectorstore(docs)
    else:
        print("📦 Laad bestaande vectorstore...")
        vectorstore = FAISS.load_local(
            "faiss_index",
            OllamaEmbeddings(model="mistral"),
            allow_dangerous_deserialization=True,
        )

    while True:
        vraag = input("Stel je vraag over de leerboeken: ")
        antwoord = ask_question(vraag, vectorstore)
        print("\n🧠 Antwoord:\n", antwoord)
