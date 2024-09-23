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

class CosmosDBEmbeddingPipeline:
    def __init__(self, source_file_name, password, db_name, collection_name, embedding_model_name,index_name,namespace, connection_string_template):
        self.source_file_name = source_file_name
        self.encoded_password = urllib.parse.quote_plus(password)
        self.db_name = db_name
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.connection_string_template = connection_string_template
        self.index_name = index_name
        self.namespace=namespace
        self.connection_string = self.connection_string_template.format(encoded_password=self.encoded_password)
        
    ##connect to cosmos db
    def connect_to_cosmos_db(self):
        try:
            client = MongoClient(self.connection_string)
            client.admin.command('ping')
            print("Connection to Cosmos DB was successful!")
            self.collection = client[self.db_name][self.collection_name]
        except Exception as e:
            print("An error occurred while connecting to Cosmos DB:", e)


    ##loading the embedding models
    def load_embeddings_model(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)


    ##load and splitting of the documents
    def load_and_split_documents(self):
        loader = PDFPlumberLoader(self.source_file_name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)


    # Add documents (chunks) to the cosmosdb vector store
    def create_vectorstore_and_index(self):
        docs=self.load_and_split_documents()
        self.connect_to_cosmos_db()
        print("Creating vector store and index...")
        start_time = time.time()

        self.vectorstore = AzureCosmosDBVectorSearch.from_documents(
            docs,
            self.load_embeddings_model(),
            collection=self.collection,
            index_name=self.index_name
        )
        # Define index parameters
        num_lists = 100
        dimensions = 1024
        similarity_algorithm = CosmosDBSimilarityType.COS  # Similarity type
        kind = CosmosDBVectorSearchType.VECTOR_IVF  # Vector search type
        m = 16
        ef_construction = 64

        self.vectorstore.create_index(
            num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
        )

        end_time = time.time()
        print(f"Time taken to store data: {end_time - start_time:.6f} seconds")


    # Retrieve documents (chunks) from the cosmosdb vector store
    def create_vectorstore_and_retrieve(self,query):
        #self.connect_to_cosmos_db()
        print("Retrieve the data based on the query...")
        start_time = time.time()
        self.vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
            self.connection_string, self.namespace, self.load_embeddings_model(), index_name=self.index_name
        )
        docs = self.vectorstore.similarity_search(query)
        print(docs[0].page_content)
        end_time=time.time()
        print(f"Time taken to retrieve data: {end_time - start_time:.6f} seconds")
        

    
SOURCE_FILE_NAME = "CV_Soumi_Karmakar_updated.pdf"
PASSWORD = "mongodb@123"
DB_NAME = "SampleDB"
COLLECTION_NAME = "embeddings"
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
INDEX_NAME="sample_index"
CONNECTION_STRING_TEMPLATE = "mongodb+srv://ikegaiuser:{encoded_password}@ikegai-cosmos-mongodb-vcore.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
NAMESPACE="SampleDB.embeddings"
query = "What are the skillset present by Soumi Karmakar?"

# Create an instance of the class
pipeline = CosmosDBEmbeddingPipeline(
        source_file_name=SOURCE_FILE_NAME,
        password=PASSWORD,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        connection_string_template=CONNECTION_STRING_TEMPLATE
    )

#store the pdf in cosmosdb
pipeline.create_vectorstore_and_index()

#retrieve the query from cosmosdb
pipeline.create_vectorstore_and_retrieve(query)

