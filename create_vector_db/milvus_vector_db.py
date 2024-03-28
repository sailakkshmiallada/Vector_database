# import os
# from llama_index import StorageContext
# from llama_index import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.vector_stores import MilvusVectorStore
#
# os.environ['OPENAI_API_KEY'] = 'your_private_key'
# os.environ['OPENAI_API_BASE'] = 'your_base'
#
#
# documents = SimpleDirectoryReader("./vectordb_pdf/").load_data()
# vector_store = MilvusVectorStore(dim=1536, overwrite=True)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author learn?")
# print(response)

from pymilvus import Milvus, MilvusException

client = Milvus(uri='tcp://localhost:19530')
try:
    status = client.has_collection("some_collection")
    print("Milvus server is running.")
except MilvusException as e:
    print(f"Failed to connect to Milvus server: {e}")
