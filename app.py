import gradio as gr
from core import BioAssistant

def chat_with_rag(message: str, history: list) -> str:
    """
    Gradio interface callback function for RAG chat.
    
    Args:
        message (str): The current user input query.
        history (list): The conversation history (managed by Gradio).
        
    Returns:
        str: The generated response appended with source citations.
    """
    assistant = BioAssistant()
    answer, sources = assistant.rag_chat(message)
    
    seen_sources = set()
    source_text = "\n\n**References:**\n"
    for doc in sources:
        src = doc.metadata.get('source', 'Unknown')
        if src not in seen_sources:
            source_text += f"- {src}\n"
            seen_sources.add(src)
            
    return answer + source_text

def main():
    """Initializes and launches the Gradio web interface."""
    demo = gr.ChatInterface(
        fn=chat_with_rag,
        title="Pharma-RAG Assistant",
        description="A Multimodal Intelligent Medical Literature Retrieval System based on DeepSeek-V3 and BGE-M3.",
        examples=[
            "第四代 EGFR TKI 的主要挑战是什么？", 
            "SCLC 转化的转录组学机制是什么？"
        ]
    )
    # Set share=True if public URL is needed
    demo.launch(share=False) 

if __name__ == "__main__":
    main()