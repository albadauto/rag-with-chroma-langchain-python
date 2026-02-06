from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
load_dotenv()
PASTA_BASE = "base"

def create_db():
    #Carrega documento
    documents = load_documents()
    #Divide em chunks (parte de textos)
    chunks = set_chunks(documents)
    #Vetorizar os chunks
    vectorize_chunks(chunks)


def load_documents():
    loader = PyPDFDirectoryLoader(PASTA_BASE)
    return loader.load()

def set_chunks(documents):
    split_text = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )
    chunks = split_text.split_documents(documents)
    return chunks

def vectorize_chunks(chunks):
    Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db")
    print("DB criada com sucesso!")

create_db()