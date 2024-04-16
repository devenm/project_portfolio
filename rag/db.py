from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import shutil
from tqdm import tqdm

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def load_docs():
    loader = TextLoader("bsb.md")
    docs = loader.load()
    return docs

def text_chunk(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents: {len(documents)} Chunks: {len(chunks)}")
    return chunks

def chroma_init_store(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    progress_chunks = tqdm(chunks, desc="Processing chunks")
    db = Chroma.from_documents(
        progress_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    
    print(f"{len(chunks)} chunks saved to {CHROMA_PATH}.")

def generate_db():
    docs = load_docs()
    chunks = text_chunk(docs)
    chroma_init_store(chunks)

def main():
    generate_db()

if __name__ == "__main__":
    main()