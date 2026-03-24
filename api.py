import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from core import BioAssistant

# Configure logging to match the core module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Pharma-RAG Enterprise API",
    description="Core backend interface for the Biomedical Literature Intelligent Retrieval System.",
    version="2.0.0"
)

# 2. Global Initialization of the RAG Assistant (Singleton Pattern)
logger.info("Initializing BioAssistant globally to prevent redundant model loading...")
assistant = BioAssistant()

# 3. Pydantic Models for Data Validation
class QueryRequest(BaseModel):
    """Schema for the incoming user query."""
    question: str = Field(..., description="The medical or biological question to ask the RAG system.")

class QueryResponse(BaseModel):
    """Schema for the structured JSON response."""
    answer: str = Field(..., description="The generated answer based on retrieved literature.")
    sources: List[str] = Field(..., description="List of unique source document paths cited in the answer.")

# 4. Expose the POST Endpoint
@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Processes a medical query through the RAG pipeline.
    
    Consumes the generator from the core RAG engine to extract the final complete answer
    and deduplicates the cited sources before returning a standardized JSON response.
    """
    logger.info(f"Received query: {request.question}")
    final_answer = ""
    final_sources = []
    
    # Consume the generator yielded by core.py to get the final output
    for partial_answer, sources in assistant.rag_chat_stream(request.question):
        final_answer = partial_answer
        final_sources = sources
        
    # Deduplicate and format source documents
    seen = set()
    source_list = []
    for doc in final_sources:
        src = doc.metadata.get('source', 'Unknown')
        if src not in seen:
            source_list.append(src)
            seen.add(src)
            
    logger.info("Successfully generated response and formatted sources.")
    
    return QueryResponse(
        answer=final_answer,
        sources=source_list
    )

# 5. Entry Point Isolation
if __name__ == "__main__":
    # Launch the ASGI server using Uvicorn
    logger.info("Starting Uvicorn server on http://127.0.0.1:8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)