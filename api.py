import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from agent_engine import PharmaAgentEngine

# 统一使用 agent_engine 中的日志配置，此处仅做基础声明
logger = logging.getLogger("PharmaAgent")

app = FastAPI(
    title="Pharma-Agent V3.0 Enterprise API",
    description="基于 LangGraph 状态机与工具调用的医疗智能体后端接口",
    version="3.0.0"
)

# 全局单例初始化 Agent 引擎
agent_engine = PharmaAgentEngine()

class AgentQueryRequest(BaseModel):
    """请求体规范：增加 session_id 用于追踪多轮对话"""
    question: str = Field(..., description="用户输入的医学问题")
    session_id: str = Field(default="default_session", description="会话唯一标识符，相同的 ID 将共享对话记忆")

class AgentQueryResponse(BaseModel):
    """响应体规范：直接返回 Agent 的综合报告"""
    answer: str = Field(..., description="Agent 经过多步推理和工具调用后生成的最终报告")
    session_id: str = Field(..., description="当前会话标识")

@app.post("/agent/chat", response_model=AgentQueryResponse)
def agent_chat_endpoint(request: AgentQueryRequest) -> AgentQueryResponse:
    """
    接收用户问题，启动 LangGraph 状态机进行自主决策、工具调用与记忆溯源，返回综合分析结果。
    """
    logger.info(f"[ROUTER] POST /agent/chat accessed by session: {request.session_id}")
    
    # 调用引擎，传入问题和会话 ID
    final_report = agent_engine.chat(question=request.question, thread_id=request.session_id)
    
    return AgentQueryResponse(
        answer=final_report,
        session_id=request.session_id
    )

if __name__ == "__main__":
    logger.info("[SYSTEM] Starting Uvicorn server on http://127.0.0.1:8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)