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
# 1. 全局日志降噪配置 (Industrial Logging)
# ==========================================
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger("PharmaAgent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('\033[92m[%(asctime)s] [%(levelname)s] %(message)s\033[0m', datefmt='%H:%M:%S')
    )
    logger.addHandler(console_handler)

# ==========================================
# 2. 状态定义与系统提示词
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = """你是一个严谨的生物医药 AI 智能体。你具备调用外部工具的能力。
请严格遵守以下纪律：
1. 必须基于工具返回的客观事实进行总结，严禁捏造医学数据。
2. 强制溯源：如果工具返回的结果中包含了“文献来源”，你在生成最终回答时，必须在末尾原封不动地附上这些文献列表。"""

# ==========================================
# 3. 核心 Agent 类封装
# ==========================================
class PharmaAgentEngine:
    def __init__(self):
        logger.info("[SYSTEM] Initializing PharmaAgentEngine and underlying RAG...")
        self.rag_engine = BioAssistant()
        self.app = self._build_graph()

    def _build_graph(self):
        # 定义内部工具
        @tool
        def calculate_drug_half_life(drug_name: str) -> str:
            """
            计算给定药物分子的预期半衰期。
            触发条件: 当用户精确询问某种药物的半衰期、代谢时间或清除时间时。
            参数: drug_name (str) 药物的通用名称。
            """
            logger.info(f"[TOOL] Executing [calculate_drug_half_life] for: {drug_name}")
            mock_db = {"奥希替尼": "约为 48 小时", "阿美替尼": "约为 35 小时"}
            return mock_db.get(drug_name, f"未命中本地药代动力学数据库: {drug_name}")

        @tool
        def search_medical_literature(query: str) -> str:
            """
            检索本地权威医药文献知识库。
            触发条件: 当用户询问复杂的医学机制、临床试验数据、疗效对比、耐药机制时，必须调用此工具。
            参数: query (str) 需要检索的具体医学问题。
            """
            logger.info(f"[TOOL] Executing [search_medical_literature] for: {query}")
            final_answer = ""
            final_sources = []
            for partial_answer, sources in self.rag_engine.rag_chat_stream(query):
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
        
        llm = ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
        )
        llm_with_tools = llm.bind_tools(tools)

        def agent_node(state: AgentState):
            logger.info("[AGENT] Evaluating state and planning next action...")
            messages = state["messages"]
            if not messages or getattr(messages[0], "type", "") != "system":
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def chat(self, question: str, thread_id: str = "default_session") -> str:
        """
        对外暴露的标准对话接口，支持传入 thread_id 以保持上下文记忆。
        """
        logger.info(f"[API] Processing request for thread: {thread_id}")
        initial_state = {"messages": [("user", question)]}
        config = {"configurable": {"thread_id": thread_id}}
        
        # 内部消费生成器流，提取最终状态
        for event in self.app.stream(initial_state, config=config):
            pass 
            
        final_state = self.app.get_state(config)
        if final_state.values.get("messages"):
            return final_state.values["messages"][-1].content
        return "System encountered an error generating a response."