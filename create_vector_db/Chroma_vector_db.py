from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import os
import chromadb
from llama_index.llms import LiteLLM

os.environ['OPENAI_API_KEY'] = 'your_private_key'
os.environ['OPENAI_API_BASE'] = 'your_base'
llm = LiteLLM("openai/Predera/llama-v2-7b-chat")


def create_chroma_vector_db(input_file, model_name="BAAI/bge-base-en-v1.5"):
    documents = SimpleDirectoryReader(input_file).load_data()
    client = chromadb.EphemeralClient()
    embeddings_model = HuggingFaceEmbedding(model_name=model_name)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings_model)
    chroma_collection = client.create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_database = VectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                                      service_context=service_context)
    return vector_database
