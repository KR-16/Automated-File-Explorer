import os
import time
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Load Documents
def load_documents(directory="docs"):
    print(f"\n[{datetime.now()}] Starting to load documents from '{directory}' directory...")
    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' does not exist!")
        return []
        
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
    }
    documents = []
    file_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[-1].lower()
            if ext in loaders:
                print(f"Loading {file}...")
                loader = loaders[ext](file_path)
                documents.extend(loader.load())
                file_count += 1
                
    print(f"[{datetime.now()}] Loaded {file_count} documents successfully")
    return documents

# 2. Split Documents into Chunks
def split_docs(documents):
    print(f"\n[{datetime.now()}] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[{datetime.now()}] Created {len(chunks)} chunks from documents")
    return chunks

# 3. Create Vector Database
def create_vector_store(docs, save_path="faiss_index"):
    print(f"\n[{datetime.now()}] Creating vector store...")
    print("This step might take a few minutes depending on the number of documents")
    
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(save_path)
    
    duration = time.time() - start_time
    print(f"[{datetime.now()}] Vector store created and saved in {duration:.2f} seconds")
    return vector_store

# 4. Initialize LLM
def load_llm():
    print(f"\n[{datetime.now()}] Loading language model...")
    model_name = "microsoft/phi-3-mini-4k-instruct"
    cache_dir = os.path.join(os.getcwd(), "models")  # Create a local models directory
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"First run: Will download model to {cache_dir}")
        print("This will download ~4GB of data and might take 5-10 minutes")
    else:
        print("Using cached model from previous run")
    
    start_time = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, model_name))
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, model_name))
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            device_map='auto'  # This will use GPU if available
        )
        
        duration = time.time() - start_time
        print(f"[{datetime.now()}] Model loaded successfully in {duration:.2f} seconds")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# 5. Query System
class DocumentExplorer:
    def __init__(self):
        print(f"\n[{datetime.now()}] Initializing Document Explorer...")
        self.llm = load_llm()
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print(f"[{datetime.now()}] Initialization complete")
        
    def build_index(self, docs_folder="docs"):
        print(f"\n[{datetime.now()}] Building search index...")
        documents = load_documents(docs_folder)
        if not documents:
            print("No documents found to index!")
            return
            
        split_documents = split_docs(documents)
        self.vector_store = create_vector_store(split_documents)
        print(f"[{datetime.now()}] Index building completed")
        
    def query(self, question):
        print(f"\n[{datetime.now()}] Processing query: {question}")
        if not self.vector_store:
            raise ValueError("Index not built. Run build_index() first.")
            
        start_time = time.time()
        
        print(f"[{datetime.now()}] Step 1: Setting up retriever...")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        print(f"[{datetime.now()}] Step 2: Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        print(f"[{datetime.now()}] Step 3: Invoking query (this might take a few minutes)...")
        try:
            result = qa_chain.invoke({"query": question})
            print(f"[{datetime.now()}] Query execution completed")
        except Exception as e:
            print(f"Error during query: {str(e)}")
            raise
        
        duration = time.time() - start_time
        print(f"[{datetime.now()}] Query processed in {duration:.2f} seconds")
        
        # Add timeout for response
        if duration > 300:  # 5 minutes timeout
            print("Query taking too long, you might want to:")
            print("1. Check if your docs folder has too many documents")
            print("2. Reduce the chunk size in split_docs()")
            print("3. Use a smaller/faster model")
        
        return {
            "answer": result["result"].strip(),
            "sources": list(set([doc.metadata["source"] for doc in result["source_documents"]])),
        }

# 6. Test with Small Dataset
if __name__ == "__main__":
    print(f"\n[{datetime.now()}] Starting Document Explorer System...")
    print("Note: First run will download required models (~4GB) and may take 10-15 minutes")
    
    start_time = time.time()
    explorer = DocumentExplorer()
    
    print("\nBuilding document index...")
    explorer.build_index()
    
    question = "Where is the API documentation?"
    print(f"\nProcessing example query: '{question}'")
    
    # Add timeout warning
    query_start = time.time()
    response = explorer.query(question)
    query_duration = time.time() - query_start
    
    if query_duration > 60:  # If query takes more than 1 minute
        print("\nNote: Query took longer than expected. This might be because:")
        print("1. The model is still loading/optimizing")
        print("2. You have many documents to search through")
        print("3. Your system is running other resource-intensive tasks")
    
    total_duration = time.time() - start_time
    print(f"\nResults:")
    print(f"Answer: {response['answer']}")
    print(f"Sources: {response['sources']}")
    print(f"\nTotal execution time: {total_duration:.2f} seconds")
