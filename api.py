import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from core import BioAssistant

# 1. 实例化 FastAPI 应用 (这是你的后端服务引擎)
app = FastAPI(
    title="Pharma-RAG Enterprise API",
    description="医药文献智能检索系统核心后端接口",
    version="2.0.0"
)

# 2. 全局加载大模型引擎 (单例模式，避免每次请求重复加载)
assistant = BioAssistant()

# 3. 定义数据验证模型 (Pydantic)
# 作用：严格规范前端传过来的数据格式，如果前端少传了字段或格式不对，FastAPI 会自动拦截并报错。
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# 4. 暴露 POST 请求接口
@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    """
    接收用户的医学问题，经过 RAG 管线处理后，返回纯 JSON 格式的答案和引文列表。
    """
    final_answer = ""
    final_sources = []
    
    # 因为我们的 core.py 改成了 yield 生成器，所以这里我们需要把它消费完，提取最终的完整结果
    for partial_answer, sources in assistant.rag_chat_stream(request.question):
        final_answer = partial_answer
        final_sources = sources
        
    # 对文献来源进行去重和格式化
    seen = set()
    source_list = []
    for doc in final_sources:
        src = doc.metadata.get('source', 'Unknown')
        if src not in seen:
            source_list.append(src)
            seen.add(src)
            
    # 返回标准的 JSON 字典
    return {
        "answer": final_answer,
        "sources": source_list
    }

# 5. 入口隔离与服务器启动
if __name__ == "__main__":
    # 使用 Uvicorn 启动 ASGI 服务器，监听本地 8000 端口
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)