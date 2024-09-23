from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import urllib.parse
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
)
import time
password = "mongodb@123"
encoded_password = urllib.parse.quote_plus(password)

EMBEDDING="mixedbread-ai/mxbai-embed-large-v1"

class Embeddings:
    def __init__(self,name) -> None:
        self.name=name
    def load(self):
        return HuggingFaceEmbeddings(model_name=self.name)
    
embeddings_model=Embeddings(EMBEDDING).load()

connection_string = f"mongodb+srv://ikegaiuser:{encoded_password}@ikegai-cosmos-mongodb-vcore.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
INDEX_NAME = "sample_index"
NAMESPACE = "SampleDB.embeddings"

client: MongoClient = MongoClient(connection_string)
db = client['SampleDB']
collection = db['embeddings']

try:
    client = MongoClient(connection_string)
    client.admin.command('ping')
    print("Connection to Cosmos DB was successful!")
except Exception as e:
    print("An error occurred:", e)

start_time=time.time()

vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
    connection_string, NAMESPACE, embeddings_model, index_name=INDEX_NAME
)


query = "From which company does Dipanwita hold an offer letter?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)
end_time=time.time()
print(f"Time taken to retrieve data: {end_time - start_time:.6f} seconds")
