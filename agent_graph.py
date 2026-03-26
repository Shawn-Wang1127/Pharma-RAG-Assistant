import os
import logging
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from core import BioAssistant

load_dotenv()

# ==========================================
# 1. 日志配置 (日志降噪与高亮)
# ==========================================
# 强制提升底层第三方库的日志拦截级别，屏蔽冗余的 INFO 刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# 自定义核心流转日志，使用 ANSI 绿色高亮显示
logger = logging.getLogger("PharmaAgent")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('\033[92m[%(asctime)s] %(message)s\033[0m', datefmt='%H:%M:%S')
)
# 清除默认的 root handlers，防止日志重复打印
logging.getLogger().handlers = []
logger.addHandler(console_handler)

# ==========================================
# 2. 核心业务引擎初始化
# ==========================================
logger.info("[SYSTEM] Initializing underlying RAG engine (BioAssistant)...")
rag_engine = BioAssistant()

# ==========================================
# 3. 定义工具集合 (Tools)
# ==========================================
@tool
def calculate_drug_half_life(drug_name: str) -> str:
    """
    计算给定药物分子的预期半衰期。
    触发条件: 当用户精确询问某种药物的半衰期、代谢时间或清除时间时。
    参数: drug_name (str) 药物的通用名称。
    """
    logger.info(f"[TOOL] Executing [calculate_drug_half_life] for argument: {drug_name}")
    mock_db = {
        "奥希替尼": "约为 48 小时",
        "阿美替尼": "约为 35 小时"
    }
    return mock_db.get(drug_name, f"未命中本地药代动力学数据库: {drug_name}")

@tool
def search_medical_literature(query: str) -> str:
    """
    检索本地权威医药文献知识库。
    触发条件: 当用户询问复杂的医学机制、临床试验数据、疗效对比、耐药机制时，必须调用此工具获取深度信息。
    参数: query (str) 需要检索的具体医学问题。
    """
    logger.info(f"[TOOL] Executing [search_medical_literature] for query: {query}")
    
    final_answer = ""
    final_sources = []
    
    for partial_answer, sources in rag_engine.rag_chat_stream(query):
        final_answer = partial_answer
        final_sources = sources
        
    seen = set()
    source_list = []
    for doc in final_sources:
        src = doc.metadata.get('source', 'Unknown')
        if src not in seen:
            source_list.append(src)
            seen.add(src)
            
    formatted_sources = "\n".join([f"- {s}" for s in source_list])
    return f"检索结论:\n{final_answer}\n\n文献来源:\n{formatted_sources}"

tools = [calculate_drug_half_life, search_medical_literature]

# ==========================================
# 4. 定义图状态 (State) 与 系统提示词 (System Prompt)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = """你是一个严谨的生物医药 AI 智能体。你具备调用外部工具的能力。
请严格遵守以下纪律：
1. 必须基于工具返回的客观事实进行总结，严禁捏造医学数据。
2. 强制溯源：如果工具返回的结果中包含了“文献来源”或“References”，你在生成最终回答时，必须在末尾原封不动地附上这些文献列表，绝不允许擅自省略。"""

# ==========================================
# 5. 初始化模型与节点 (Nodes)
# ==========================================
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
)

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    logger.info("[AGENT] Node invoked. LLM is evaluating state and planning next action...")
    messages = state["messages"]
    
    # 动态注入系统提示词，确保其始终在对话历史的最前方发挥约束作用
    if not messages or getattr(messages[0], "type", "") != "system":
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ==========================================
# 6. 构建并编译状态图 (Graph)
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==========================================
# 7. 交互式多轮对话终端 (Interactive CLI)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("[SYSTEM] Pharma-Agent V3.0 CLI 已启动")
    print("[SYSTEM] 状态: 就绪 | 记忆模块: 开启 (thread_id: session_01)")
    print("[SYSTEM] 输入 'quit' 或 'exit' 终止会话")
    print("="*60 + "\n")
    
    # 核心：固定的 thread_id 保证了上下文在 while 循环中不会丢失
    config = {"configurable": {"thread_id": "session_01"}}
    
    while True:
        try:
            user_input = input("\n[USER] 请输入指令: ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("[SYSTEM] 会话已安全终止。")
                break
            if not user_input.strip():
                continue
                
            # 将用户新输入推入状态图
            initial_state = {"messages": [("user", user_input)]}
            
            print("\n" + "-"*60)
            # 触发图的执行
            for event in app.stream(initial_state, config=config):
                for key, value in event.items():
                    logger.info(f"[SYSTEM] Node completed: [{key}]")
                    
            # 提取并打印本轮的最终回复
            final_state = app.get_state(config)
            if final_state.values.get("messages"):
                final_message = final_state.values["messages"][-1].content
                print("\n[AGENT OUTPUT]")
                print(final_message)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n[SYSTEM] 接收到中断信号，强制退出。")
            break