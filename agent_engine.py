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

SYSTEM_PROMPT = """你是一个严谨的生物医药 AI 智能体。你具备调度多个专业检索工具与代码沙盒的能力。
请严格遵守以下纪律：
1. 双引擎协同：本地医学知识库 (search_medical_literature) 与 PubMed (search_pubmed_literature) 视需求调度。
2. 数据分析双步法则（核心纪律）：当需要分析 CSV 数据时，你必须遵守两步走战略：
   - 第一步：必须先调用 preview_csv_data 获取真实的列名和数据样本。
   - 第二步：根据上一步获取的真实列名，调用 execute_python_code 编写 pandas 脚本进行计算。
   绝不允许在没有探查真实列名的情况下凭空猜测并编写数据分析代码。
3. 可视化：若需画图，必须在代码开头写 `import matplotlib; matplotlib.use('Agg')` 并保存为本地图片。
4. 强制溯源：不捏造数据，引用文献须附上来源或 PMID。
5. 可视化排版纪律(防重叠)：若需生成带有统计信息文本框(bbox)的柱状图,代码必须具备动态布局意识：
   - 第一选择：严禁使用绝对坐标定位文本框。应使用 `fig.subplots()` 创建双子图排版，将统计文本专门放置在左侧或右侧的子图中，将图表放置在另一侧。
   - 第二选择（若必须在单图中显示）：代码必须动态计算数据最大值（如 `max(means) * 1.5`），并将 `plt.ylim` 设置得足够高，为顶部的文本框预留出至少 30% 的空白缓冲区，避免文字标签与 bbox 重叠。"""

# ==========================================
# 3. 核心 Agent 类封装
# ==========================================
class PharmaAgentEngine:
    def __init__(self):
        logger.info("[SYSTEM] Initializing PharmaAgentEngine and underlying RAG...")
        self.rag_engine = BioAssistant()
        self.app = self._build_graph()

    def _build_graph(self):
        # ----------------------------------------
        # 工具 1：本地高密级 RAG 检索器
        # ----------------------------------------
        @tool
        def search_medical_literature(query: str) -> str:
            """
            检索本地权威医药文献知识库。
            触发条件: 当用户询问复杂的医学机制、已知临床试验数据（如 MARIPOSA 试验）、疗效对比时调用。
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
            return f"本地检索结论:\n{final_answer}\n\n文献来源:\n{formatted_sources}"

        # ----------------------------------------
        # 工具 2：NCBI PubMed 全网文献抓取器 (新增)
        # ----------------------------------------
        @tool
        def search_pubmed_literature(query: str, max_results: int = 3) -> str:
            """
            检索 NCBI PubMed 全球公开医学文献数据库。
            触发条件: 当用户要求查询最新的全球研究进展、特定靶点的最新发表文献，或本地库未命中时调用。
            参数: 
                query (str) 符合 PubMed 搜索规范的英文检索词 (例如 "EGFR mutation AND non-small cell lung cancer")。
                max_results (int) 需要抓取的最新文献数量，默认 3 篇。
            """
            from Bio import Entrez, Medline
            import io
            import ssl
            
            # NCBI 要求提供开发者邮箱以限制恶意请求
            Entrez.email = "pharma_agent_developer@example.com"
            # 规避部分本地环境的 SSL 证书验证问题
            ssl._create_default_https_context = ssl._create_unverified_context
            
            logger.info(f"[TOOL] Executing [search_pubmed_literature] for query: {query}")
            
            try:
                # 1. 检索符合条件的文献 PMID (按发表时间倒序)
                handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="pub date")
                record = Entrez.read(handle)
                handle.close()
                
                id_list = record["IdList"]
                if not id_list:
                    return f"PubMed 未检索到关于 '{query}' 的相关文献，请尝试更换更广泛的英文检索词。"
                    
                # 2. 根据 PMID 抓取详细摘要数据
                fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
                medline_data = fetch_handle.read()
                fetch_handle.close()
                
                # 3. 解析结构化数据并返回给大模型
                records = Medline.parse(io.StringIO(medline_data))
                results = []
                for rec in records:
                    title = rec.get("TI", "No Title")
                    abstract = rec.get("AB", "No Abstract available.")
                    pmid = rec.get("PMID", "Unknown")
                    results.append(f"PMID: {pmid}\n标题: {title}\n摘要: {abstract}\n---")
                    
                return "\n".join(results)
                
            except Exception as e:
                logger.error(f"[TOOL] PubMed search failed: {str(e)}")
                return f"PubMed API 调用失败，请检查网络或检索词: {str(e)}"
        
        # ----------------------------------------
        # 工具 3：CSV 数据结构探查器 (新增)
        # ----------------------------------------
        @tool
        def preview_csv_data(filepath: str) -> str:
            """
            获取本地 CSV 文件的表结构（列名、数据类型）和前3行数据样本。
            触发条件: 在使用 execute_python_code 对任何 CSV 进行计算分析之前，必须【优先】调用此工具了解真实的列名，绝不允许凭空猜测列名。
            参数: filepath (str) CSV 文件的相对路径。
            """
            import pandas as pd
            logger.info(f"[TOOL] Executing [preview_csv_data] for: {filepath}")
            try:
                df = pd.read_csv(filepath)
                info = f"【文件概况】 {filepath}\n"
                info += f"总行数: {len(df)} | 总列数: {len(df.columns)}\n\n"
                info += "【真实列名及数据类型】\n"
                info += df.dtypes.to_string() + "\n\n"
                info += "【前3行数据样本】\n"
                info += df.head(3).to_string()
                return info
            except Exception as e:
                return f"探查文件失败，请检查路径: {str(e)}"
            
        # ----------------------------------------
        # 工具 4：Python 数据分析沙盒 (新增)
        # ----------------------------------------
        @tool
        def execute_python_code(code: str) -> str:
            """
            执行 Python 代码来进行本地临床数据分析。
            触发条件: 当用户要求统计数据、分析 CSV 表格或计算生存期 (OS/PFS) 等数学逻辑时调用。
            参数: code (str) 需要执行的 Python 代码段。
            """
            import sys
            import io
            import traceback
            import pandas as pd
            import numpy as np
            
            logger.info("[TOOL] Executing [execute_python_code] for dynamic data analysis...")
            logger.info(f"[TOOL] Code payload:\n{code}")
            
            # 劫持标准输出，捕获大模型代码中的 print() 结果
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output
            
            try:
                # 定义安全的局部命名空间，预加载数据分析核心库
                local_vars = {'pd': pd, 'np': np}
                
                # 执行大模型实时生成的代码
                exec(code, {}, local_vars)
                
                # 提取执行输出
                output = redirected_output.getvalue()
                if not output.strip():
                    return "代码执行成功，但未捕获到输出。请确保在代码中使用了 print() 函数打印最终结果。"
                return f"执行成功。终端输出结果:\n{output}"
                
            except Exception as e:
                error_msg = traceback.format_exc()
                logger.error(f"[TOOL] Code execution failed: {str(e)}")
                # 极其关键：将报错信息原路返回给大模型，触发其“自我修复 (Self-Correction)”机制
                return f"代码执行出错，请修正代码并重试。报错堆栈:\n{error_msg}"
                
            finally:
                # 恢复系统标准输出
                sys.stdout = old_stdout
        # 注册全栈工具箱
        tools = [search_medical_literature, search_pubmed_literature, preview_csv_data, execute_python_code]
        
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
        logger.info(f"[API] Processing request for thread: {thread_id}")
        initial_state = {"messages": [("user", question)]}
        config = {"configurable": {"thread_id": thread_id}}
        
        for event in self.app.stream(initial_state, config=config):
            pass 
            
        final_state = self.app.get_state(config)
        if final_state.values.get("messages"):
            return final_state.values["messages"][-1].content
        return "System encountered an error generating a response."

# ==========================================
# 4. 内部交互式 CLI 测试入口
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("[SYSTEM] Pharma-Agent V3.1 (NCBI PubMed Integration)")
    print("[SYSTEM] 引擎已加载: 1. 本地 RAG 知识库  2. 公网 PubMed 抓取器")
    print("="*60 + "\n")
    
    engine = PharmaAgentEngine()
    
    while True:
        try:
            user_input = input("\n[USER] 请输入指令 (输入 quit 退出): ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue
                
            response = engine.chat(question=user_input, thread_id="test_pubmed_session")
            print("\n[AGENT OUTPUT]")
            print("-" * 60)
            print(response)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n[SYSTEM] 接收到中断信号，退出。")
            break