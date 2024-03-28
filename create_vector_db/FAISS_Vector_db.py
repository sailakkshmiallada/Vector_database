from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.embeddings import HuggingFaceEmbeddings
import sys


def create_vector_db(input_file, model_name="BAAI/bge-base-en-v1.5", chunk_size=500, chunk_overlap=0):
    # Initialize embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    # Load raw documents from PDF
    raw_documents = PDFPlumberLoader(input_file).load()
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(raw_documents)
    # Create vector database using FAISS
    vector_db = FAISS.from_documents(documents, embeddings_model)
    print(sys.getsizeof(vector_db))
    return vector_db
