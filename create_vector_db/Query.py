from create_vector_db.Qdrant_vector_db import create_qdrant_vector_db
from create_vector_db.FAISS_Vector_db import create_vector_db
from create_vector_db.Chroma_vector_db import create_chroma_vector_db
from create_vector_db.Weaviate_vector_db import create_weaviate_vector_db, auth_config


def chroma_query(input_file):
    vector_database = create_chroma_vector_db(input_file)
    query_engine = vector_database.as_query_engine()
    query = input("Enter your query: ")
    response = query_engine.query(query)
    return response


def faiss_query(input_file):
    vector_database = create_vector_db(input_file)
    query = input("Enter your query: ")
    docs = vector_database.similarity_search(query)
    return docs[0].page_content


def qdrant_query(input_file):
    vector_database = create_qdrant_vector_db(input_file)
    query_engine = vector_database.as_query_engine()
    query = input("Enter your query: ")
    response = query_engine.query(query)
    return response


def weaviate_query(input_file):
    vector_database = create_weaviate_vector_db(input_file, auth_config)
    query_engine = vector_database.as_query_engine()
    query = input("Enter your query: ")
    response = query_engine.query(query)
    print(response)


def selecting_vector_db(Vector_database):
    global response, input_file

    if Vector_database == "chroma":
        input_file = "vectordb_pdf/"
        response = chroma_query(input_file)

    elif Vector_database == "faiss":
        input_file = "vectordb_pdf/vectordb.pdf"
        response = faiss_query(input_file)

    elif Vector_database == "qdrant":
        input_file = "create_vector_db/vectordb_pdf"
        response = qdrant_query(input_file)

    elif Vector_database == 'weaviate':
        input_file = "./vectordb_pdf/"
        response = weaviate_query(input_file)

    return response
