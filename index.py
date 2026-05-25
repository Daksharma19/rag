from pathlib import Path 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).parent / "WritingOS.pdf"

# Load this file in the current program 
loader = PyPDFLoader(file_path=pdf_path)
# load page by page 
docs = loader.load()


# Split the docs in smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)

chunks = text_splitter.split_documents(documents=docs)

# Vector Embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    url="http://localhost:6333/",
    collection_name="learning_rag"
)

print("Indexing of Document Done......")





# print(len(docs))

# print(chunks[100].page_content)