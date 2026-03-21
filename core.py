import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class BioAssistant:
    """
    A Retrieval-Augmented Generation (RAG) assistant tailored for biomedical literature.
    Integrates DeepSeek-V3 for generation and BGE-M3 for dense retrieval.
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
            max_tokens=2048
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,    
            chunk_overlap=300,  
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        logger.info("Initializing BGE-M3 Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )   
        
        self.persist_directory = "./chroma_db_bge_m3"
        self.vector_db = None

    def build_vector_db(self, data_folder: str) -> None:
        """
        Scans PDF documents, applies chunking, and builds the Chroma vector database.
        """
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
        if not pdf_files:
            logger.error("No PDF files found in the specified data folder.")
            return
            
        all_chunks = []
        for file_name in pdf_files:
            logger.info(f"Processing document: {file_name}")
            loader = PyPDFLoader(os.path.join(data_folder, file_name))
            all_chunks.extend(self.text_splitter.split_documents(loader.load()))
        
        logger.info(f"Vectorizing and persisting {len(all_chunks)} chunks. This may take a while...")
        self.vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.info(f"Vector database successfully built at: {self.persist_directory}")

    def query_knowledge(self, english_query: str):
        """
        Executes Maximal Marginal Relevance (MMR) search to retrieve diverse and relevant chunks.
        """
        if not os.path.exists(self.persist_directory):
            logger.warning("Local database not found. Please run build_vector_db first.")
            return []
            
        if self.vector_db is None:
            self.vector_db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
        
        # MMR Search: Fetch 30 documents initially, select 10 diverse documents
        docs = self.vector_db.max_marginal_relevance_search(
            english_query, 
            k=10,            
            fetch_k=30,      
            lambda_mult=0.5  
        )
        return docs
    
    def rag_chat(self, question: str):
        """
        Full RAG pipeline: Query Translation -> MMR Retrieval -> Augmented Generation.
        """
        logger.info("Step 1: Executing Query Translation...")
        translation_prompt = (
            "You are a medical expert. Translate the following Chinese question into "
            "professional English search keywords. Output ONLY the English keywords:\n"
            f"{question}"
        )
        english_query = self.llm.invoke(translation_prompt).content.strip()
        logger.info(f"Optimized Search Query: {english_query}")

        logger.info("Step 2: Retrieving context from vector database...")
        context_docs = self.query_knowledge(english_query)
        if not context_docs:
            return "Failed to retrieve relevant medical literature.", []

        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt_template = ChatPromptTemplate.from_template("""
        你是一位精通生物信息学和临床肿瘤学的资深助手。请基于提供的【参考资料】回答问题。

        ### 回答准则：
        1. **证据优先**：优先引用指南(Guideline)和药物标签(Label)中的数据。
        2. **深度总结**：如果资料未直接给出结论，请根据文中描述的耐药机制、临床表现进行专业归纳。
        3. **溯源规范**：请用中文回答，条理清晰，并在回答中提及参考的来源。

        ### 参考资料库:
        {context}
        
        ### 用户问题: 
        {question}
        
        请提供专业的深度解析：
        """)
        
        logger.info("Step 3: Generating augmented response via LLM...")
        chain = prompt_template | self.llm
        response = chain.invoke({"context": context_text, "question": question})
        
        return response.content, context_docs