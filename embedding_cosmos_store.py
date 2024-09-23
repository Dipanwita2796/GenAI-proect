from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from pymongo import MongoClient
import urllib.parse
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from langchain_text_splitters import CharacterTextSplitter
import time
password = "mongodb@123"
encoded_password = urllib.parse.quote_plus(password)

SOURCE_FILE_NAME = "Dipanwita_ghosh_offer_letter.pdf"


EMBEDDING="mixedbread-ai/mxbai-embed-large-v1"

class Embeddings:
    def __init__(self,name) -> None:
        self.name=name
    def load(self):
        return HuggingFaceEmbeddings(model_name=self.name)
    
embeddings_model=Embeddings(EMBEDDING).load()

loader = PDFPlumberLoader(SOURCE_FILE_NAME)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
#print(docs)

connection_string = f"mongodb+srv://ikegaiuser:{encoded_password}@ikegai-cosmos-mongodb-vcore.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
#print(connection_string)
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

vectorstore = AzureCosmosDBVectorSearch.from_documents(
    docs,
    embeddings_model,
    collection=collection,
    index_name=INDEX_NAME
)

num_lists = 100
dimensions = 1024
similarity_algorithm = CosmosDBSimilarityType.COS  # Should be an appropriate value from CosmosDBSimilarityType
kind = CosmosDBVectorSearchType.VECTOR_IVF  # Should be an appropriate value from CosmosDBVectorSearchType
m = 16
ef_construction = 64
ef_search = 40 
score_threshold = 0.1

vectorstore.create_index(
    num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
)

end_time=time.time()
print(f"Time taken to store data: {end_time - start_time:.6f} seconds")


