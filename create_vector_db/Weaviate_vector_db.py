import weaviate
import os
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms import LiteLLM
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext

os.environ['OPENAI_API_KEY'] = 'your_private_key'
os.environ['OPENAI_API_BASE'] = 'your_base'
llm = LiteLLM("openai/Predera/llama-v2-7b-chat")
auth_config = weaviate.AuthApiKey(api_key="your_api_key")


def create_weaviate_vector_db(input_file, auth_config, model_name="BAAI/bge-base-en-v1.5"):
    documents = SimpleDirectoryReader(input_file).load_data()
    client = weaviate.Client(url="your_weaviate_url", auth_client_secret=auth_config)
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings_model)
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="LlamaIndex")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_database = VectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                                      service_context=service_context)
    return vector_database



