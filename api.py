import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from agent_engine import PharmaAgentEngine

# ==========================================
# 1. 全局日志配置
# ==========================================
logger = logging.getLogger("PharmaAgent")

# ==========================================
# 2. FastAPI 实例初始化与静态资源挂载
# ==========================================
app = FastAPI(
    title="Pharma-Agent V3.1 Enterprise API",
    description="基于 LangGraph 的医疗多智能体系统，支持本地 RAG、全网文献检索与自动化数据分析",
    version="3.1.0"
)

# 核心升级：确保存放生成图表的目录存在，并将其挂载为静态资源路由
os.makedirs("clinical_data", exist_ok=True)
app.mount("/clinical_data", StaticFiles(directory="clinical_data"), name="clinical_data")

# 全局单例初始化 Agent 引擎
agent_engine = PharmaAgentEngine()

# ==========================================
# 3. 数据模型定义 (Pydantic)
# ==========================================
class AgentQueryRequest(BaseModel):
    """请求体规范：包含用户问题与会话隔离 ID"""
    question: str = Field(..., description="用户输入的医学问题或数据分析指令")
    session_id: str = Field(default="default_session", description="会话唯一标识符，相同的 ID 将共享对话记忆")

class AgentQueryResponse(BaseModel):
    """响应体规范：返回 Agent 的综合报告"""
    answer: str = Field(..., description="Agent 经过多步推理、工具调用（含图表生成）后返回的最终 Markdown 报告")
    session_id: str = Field(..., description="当前会话标识")

# ==========================================
# 4. 核心路由 Endpoint
# ==========================================
@app.post("/agent/chat", response_model=AgentQueryResponse)
def agent_chat_endpoint(request: AgentQueryRequest) -> AgentQueryResponse:
    """
    接收用户问题，启动 LangGraph 状态机。
    支持触发: 本地文献检索 (RAG), PubMed 全网检索, Python 数据分析与可视化。
    """
    logger.info(f"[ROUTER] POST /agent/chat accessed by session: {request.session_id}")
    
    # 调用底层引擎，传入问题和会话 ID
    final_report = agent_engine.chat(question=request.question, thread_id=request.session_id)
    
    return AgentQueryResponse(
        answer=final_report,
        session_id=request.session_id
    )

if __name__ == "__main__":
    logger.info("[SYSTEM] Starting Uvicorn server on http://127.0.0.1:8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)