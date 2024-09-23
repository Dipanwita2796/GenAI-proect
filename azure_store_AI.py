from langchain_community.vectorstores import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
import time

AZURE_AI_SEARCH_SERVICE_NAME="https://ikegai-ai-search.search.windows.net"
#SAZURE_AI_SEARCH_INDEX_NAME="sample_vectore_index"
AZURE_AI_SEARCH_API_KEY="2VkJo7jqIJ9aehJ38ZNmPQlYNa34yDxyZq2lJevLYWAzSeBvYB7Z"


# Define the embeddings model class
class Embeddings:
    def __init__(self, name) -> None:
        self.name = name
    
    def load(self):
        return HuggingFaceEmbeddings(model_name=self.name)
    
# Load the embeddings model
EMBEDDING = "mixedbread-ai/mxbai-embed-large-v1"
embeddings_model = Embeddings(EMBEDDING).load()



# Function to load and chunk the PDF
def load_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    # Step 1: Load PDF using LangChain's PDFPlumberLoader (community loader)
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split text into chunks using CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    #print(chunks)
    
    return chunks

start_time=time.time()

vector_store: AzureSearch=AzureSearch(
    embedding_function=embeddings_model.embed_query,
    azure_search_endpoint=AZURE_AI_SEARCH_SERVICE_NAME,
    azure_search_key=AZURE_AI_SEARCH_API_KEY,
    index_name="langchain-vector-demo",
)

pdf_path="Dipanwita_ghosh_offer_letter.pdf"
chunks_res=load_and_chunk_pdf(pdf_path)
vector_store.add_documents(documents=chunks_res)

end_time=time.time()
print(f"Time taken to store data: {end_time - start_time:.6f} seconds")


#vector_store.close()