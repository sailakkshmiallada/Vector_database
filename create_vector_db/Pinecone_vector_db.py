from langchain.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from getpass import getpass
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Initialize OpenAI
os.environ['OPENAI_API_KEY'] = 'your_private_key'
os.environ['OPENAI_API_BASE'] = 'your_base'

loader = PyPDFLoader("./vectordb_pdf/vectordb.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(data)
# print(f'Now you have {len(texts)} documents')

embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# find API key in console at app.pinecone.io
YOUR_API_KEY = 'you_api_key'
print("YOUR_API_KEY", YOUR_API_KEY)
# find ENV (cloud region) next to API key in console
YOUR_ENV = "gcp-starter"
print("YOUR_ENV", YOUR_ENV)
index_name = 'my_index'

# Now do stuff
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name='my_index',
        dimension=128,
        metric='euclidean', )
    print('Index created successfully.')
else:
    print('Index already exists.')

docsearch = Pinecone.from_documents(texts, embeddings_model, index_name=index_name)
#
# llm = ChatOpenAI(model_name="Predera/llama-v2-7b-chat")
# chain = load_qa_chain(llm, chain_type="stuff")
# query = "What is vector database and it's components"
# docs = docsearch.similarity_search(query)
# chain.run(input_documents=docs, question=query)
