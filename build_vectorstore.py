import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Set your API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 1. Load the documents
loader = DirectoryLoader(
    "./knowledge_base/", 
    glob="**/*.txt", 
    loader_cls=TextLoader,
    show_progress=True
)
docs = loader.load()

# 2. Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# 3. Create embeddings
# We use Google's free embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") 

# 4. Create and save the FAISS vector store
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("faiss_index")

print("✅ Vector store created and saved to 'faiss_index'")