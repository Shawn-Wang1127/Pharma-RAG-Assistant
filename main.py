import os
import logging
from core import BioAssistant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_evaluation_pipeline():
    """
    Entry point for CLI evaluation of the Pharma-RAG system.
    Handles automated DB instantiation and executes a benchmark query.
    """
    assistant = BioAssistant()
    
    db_path = "./chroma_db_bge_m3"
    data_folder = "data"
    
    # Ensure Vector DB is initialized
    if not os.path.exists(db_path):
        logger.info("First run detected. Initializing Vector Database indexing pipeline...")
        assistant.build_vector_db(data_folder)
    else:
        file_count = len([f for f in os.listdir(data_folder) if f.endswith('.pdf')])
        logger.info(f"Local Knowledge Base loaded. Current index covers {file_count} medical documents.")

    # Benchmark Query
    query = "根据 MARIPOSA 试验的相关文献,Amivantamab 联合 lazertinib 相比于奥希替尼单药，在晚期 EGFR 突变非小细胞肺癌中的疗效（如无进展生存期 PFS 或总生存期 OS)表现如何？"
    
    logger.info("Initiating RAG evaluation...")
    final_answer = ""
    final_sources = []
    for partial_answer, sources in assistant.rag_chat_stream(query):
        final_answer = partial_answer
        final_sources = sources

    # Formatting output
    print("\n" + "="*60)
    print(f"[User Query]: {query}")
    print("-" * 60)
    print(f"[AI Response]:\n{final_answer}")
    print("-" * 60)
    print("[Source References]:")
    
    seen_sources = set()
    for doc in final_sources:
        source_name = doc.metadata.get('source', 'Unknown')
        if source_name not in seen_sources:
            print(f" - {source_name}")
            seen_sources.add(source_name)
    print("="*60)

if __name__ == "__main__":
    run_evaluation_pipeline()