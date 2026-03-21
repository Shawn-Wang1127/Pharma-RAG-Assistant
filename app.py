import gradio as gr
from core import BioAssistant

# 1. 初始化你的底层系统
assistant = BioAssistant()
# 确保数据库已经构建完毕
assistant.query_knowledge("test") # 预热一下

# 2. 定义包装函数供网页调用
def chat_with_rag(message, history):
    # message 是用户在网页输入的新问题
    answer, sources = assistant.rag_chat(message)
    
    # 提取文献来源名字用于展示
    seen_sources = set()
    source_text = "\n\n**参考来源：**\n"
    for doc in sources:
        src = doc.metadata.get('source', 'Unknown')
        if src not in seen_sources:
            source_text += f"- {src}\n"
            seen_sources.add(src)
            
    return answer + source_text

# 3. 构建并启动网页界面
demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="医药文献智能检索系统 (Pharma RAG Assistant)",
    description="基于 DeepSeek-V3 与 BGE-M3 的专业生信/临床医学知识库",
    examples=["第四代 EGFR TKI 的主要挑战是什么？", "SCLC 转化的转录组学机制是什么？"]
)

if __name__ == "__main__":
    demo.launch() # 运行后会给你一个本地网页链接，比如 http://127.0.0.1:7860