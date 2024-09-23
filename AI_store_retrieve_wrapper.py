from langchain_community.vectorstores import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
import time

class PDFVectorStore:
    def __init__(self, service_name, api_key, index_name, embedding_model_name):
        self.service_name = service_name
        self.api_key = api_key
        self.index_name = index_name
        self.embedding_model_name = embedding_model_name
        self.embeddings_model = self.load_embeddings_model()
        self.vector_store = self.init_vector_store()

    # Load the embeddings model
    def load_embeddings_model(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
    

    # Load and chunk the PDF
    def load_and_chunk_pdf(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        return chunks
    

    # Initialize AzureSearch Vector Store
    def init_vector_store(self):
        return AzureSearch(
            embedding_function=self.embeddings_model.embed_query,
            azure_search_endpoint=self.service_name,
            azure_search_key=self.api_key,
            index_name=self.index_name
        )

    
    # Add documents (chunks) to the AI Search vector store
    def store_pdf_in_vector_store(self, pdf_path):
        start_time = time.time()
        
        # Load and chunk the PDF
        chunks = self.load_and_chunk_pdf(pdf_path)

        # Add chunks to the vector store
        self.vector_store.add_documents(documents=chunks)

        end_time = time.time()
        print(f"Time taken to store data: {end_time - start_time:.6f} seconds")


    # Retrieve documents (chunks) from the vector store
    def retrieve_from_vector_store(self,query):
        start_time = time.time()
        search_results = self.vector_store.similarity_search(query)
        print(search_results[0].page_content)
        end_time=time.time()
        print(f"Time taken to retrieve data: {end_time - start_time:.6f} seconds")


#AZURE_AI_SEARCH_SERVICE_NAME = "https://ikegai-ai-search.search.windows.net"
#AZURE_AI_SEARCH_API_KEY = "2VkJo7jqIJ9aehJ38ZNmPQlYNa34yDxyZq2lJevLYWAzSeBvYB7Z"
INDEX_NAME = "langchain-vector-demo"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
PDF_PATH = "Dipanwita Ghosh_WORD_RESUME_final_mlops.pdf"
query="From which company did dipanwita got an offer?"

# Create an instance of the class
pdf_vector_store = PDFVectorStore(
    service_name=AZURE_AI_SEARCH_SERVICE_NAME,
    api_key=AZURE_AI_SEARCH_API_KEY,
    index_name=INDEX_NAME,
    embedding_model_name=EMBEDDING_MODEL
)

# Store the PDF in the AI vector store
pdf_vector_store.store_pdf_in_vector_store(PDF_PATH)

#Retrieve the embeddings from AI vector store
pdf_vector_store.retrieve_from_vector_store(query)
