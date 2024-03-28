import qdrant_client
import os
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms import LiteLLM
import sys

os.environ['OPENAI_API_KEY'] = 'your_private_key'
os.environ['OPENAI_API_BASE'] = 'your_base'
llm = LiteLLM("openai/Predera/llama-v2-7b-chat")


def create_qdrant_vector_db(input_file, model_name="BAAI/bge-base-en-v1.5"):
    documents = SimpleDirectoryReader(input_file).load_data()
    client = qdrant_client.QdrantClient(location=":memory:")
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings_model)
    vector_store = QdrantVectorStore(client=client, collection_name="vector_db")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_database = VectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                                      service_context=service_context)
    print(sys.getsizeof(vector_database))
    return vector_database
