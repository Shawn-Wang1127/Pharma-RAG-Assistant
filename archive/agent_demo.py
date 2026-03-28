import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# ==========================================
# 1. 打造工具 (Tools)
# ==========================================
@tool
def calculate_drug_half_life(drug_name: str) -> str:
    """
    计算给定药物分子的预期半衰期。
    当用户询问药物的半衰期、代谢时间时，必须调用此工具。
    """
    mock_db = {
        "奥希替尼": "约为 48 小时",
        "阿美替尼": "约为 35 小时"
    }
    return mock_db.get(drug_name, f"外部数据库中未找到 {drug_name} 的半衰期数据。")

# 建立工具映射字典，方便后续调用
tool_map = {"calculate_drug_half_life": calculate_drug_half_life}

# ==========================================
# 2. 初始化大模型并“装配”工具
# ==========================================
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
)
tools = [calculate_drug_half_life]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 完整的 Agent 执行循环
# ==========================================
if __name__ == "__main__":
    query = "请问奥希替尼的半衰期是多久？"
    print(f"👨‍⚕️ 用户问题: {query}\n")
    
    # 将用户问题转化为标准的消息格式
    messages = [HumanMessage(content=query)]
    
    # 【第一轮对话】：大模型思考并决定是否调用工具
    print("🤖 [Step 1: LLM 思考中...]")
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg) # 把 LLM 的回复（包含工具指令）加入对话历史
    
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"🛠️ [Step 2: 触发工具调用] LLM 请求执行: {tool_call['name']}")
            print(f"   传入参数: {tool_call['args']}")
            
            # 系统真实执行 Python 函数
            selected_tool = tool_map[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            print(f"   工具执行结果: {tool_output}\n")
            
            # 将工具执行的结果打包成 ToolMessage，加回对话历史
            tool_msg = ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
            messages.append(tool_msg)
            
        # 【第二轮对话】：把携带了工具运行结果的历史记录，再次喂给大模型
        print("🤖 [Step 3: LLM 吸收工具结果，生成最终回复...]")
        final_response = llm_with_tools.invoke(messages)
        print(f"\n✅ 最终回答: {final_response.content}")
    else:
        # 如果大模型觉得不需要用工具，就直接输出文本
        print(f"\n✅ 最终回答: {ai_msg.content}")