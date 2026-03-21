import gradio as gr
from core import BioAssistant

# Initialize the RAG assistant globally to avoid reloading models on every chat turn
assistant = BioAssistant()

def chat_with_rag_stream(message: str, history: list):
    """
    Gradio interface generator function for streaming RAG chat.
    
    Args:
        message (str): The current user input query.
        history (list): The conversation history (managed by Gradio).
        
    Yields:
        str: The progressively generated response appended with source citations.
    """
    # Consume the generator yielded by core.py
    for partial_answer, sources in assistant.rag_chat_stream(message):
        
        # Extract and format unique source citations
        seen_sources = set()
        source_text = "\n\n**References:**\n"
        for doc in sources:
            src = doc.metadata.get('source', 'Unknown')
            if src not in seen_sources:
                source_text += f"- {src}\n"
                seen_sources.add(src)
                
        # Yield the combined string progressively to update the UI
        yield partial_answer + source_text

def main():
    """Initializes and launches the Gradio web interface with streaming support."""
    demo = gr.ChatInterface(
        fn=chat_with_rag_stream,
        title="Pharma-RAG Assistant",
        description="A Multimodal Intelligent Medical Literature Retrieval System based on DeepSeek-V3 and BGE-M3. (Streaming Enabled)",
        examples=[
            "第四代 EGFR TKI 的主要挑战是什么？", 
            "SCLC 转化的转录组学机制是什么？"
        ]
    )
    # Set share=True if public URL is needed
    demo.launch(share=False) 

if __name__ == "__main__":
    main()