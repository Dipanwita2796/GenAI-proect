from langchain_community.vectorstores import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
import time

#AZURE_AI_SEARCH_SERVICE_NAME="https://ikegai-ai-search.search.windows.net"
#SAZURE_AI_SEARCH_INDEX_NAME="sample_vectore_index"
#AZURE_AI_SEARCH_API_KEY="2VkJo7jqIJ9aehJ38ZNmPQlYNa34yDxyZq2lJevLYWAzSeBvYB7Z"


# Define the embeddings model class
class Embeddings:
    def __init__(self, name) -> None:
        self.name = name
    
    def load(self):
        return HuggingFaceEmbeddings(model_name=self.name)
    
# Load the embeddings model
EMBEDDING = "mixedbread-ai/mxbai-embed-large-v1"
embeddings_model = Embeddings(EMBEDDING).load()


start_time=time.time()

vector_store: AzureSearch=AzureSearch(
    embedding_function=embeddings_model.embed_query,
    azure_search_endpoint=AZURE_AI_SEARCH_SERVICE_NAME,
    azure_search_key=AZURE_AI_SEARCH_API_KEY,
    index_name="langchain-vector-demo",
)


query = "From which company did dipanwita got an offer?"


search_results = vector_store.similarity_search(query)
print(search_results[0].page_content)






end_time=time.time()
print(f"Time taken to retrieve data: {end_time - start_time:.6f} seconds")