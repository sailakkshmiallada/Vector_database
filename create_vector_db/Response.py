from create_vector_db.Query import selecting_vector_db

Vector_database = input("Select your Vector Database:")
response = selecting_vector_db(Vector_database)
print(response)