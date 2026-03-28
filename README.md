# Pharma-Agent (原 Pharma-RAG Assistant)

基于 LangGraph 状态机构建的医疗医药多智能体（Multi-Agent）系统。本项目从基础的 RAG 架构全面升级，具备自主决策、外部 API 调度、代码沙盒执行、多租户并发记忆跟踪以及完整的 RESTful API 接入能力。

## 💡 核心架构与三引擎 (Architecture)

* **Agent 编排框架**: LangGraph (支持 ReAct 路由、状态机管理与 MemorySaver 持久化记忆)
* **LLM 引擎**: DeepSeek-V3 (兼容 OpenAI API 规范，完美支持 Tool Calling 与错误反思机制)
* **后端服务框架**: FastAPI + Uvicorn + Pydantic (原生支持静态图表资源路由)
* **三位一体工具箱 (Tools)**:
  1. `search_medical_literature`: 本地高保密级别 RAG 检索器 (基于 BAAI/bge-m3 与 ChromaDB)。
  2. `search_pubmed_literature`: 实时接入 NCBI PubMed API，抓取全球最新医药文献摘要。
  3. `execute_python_code`: 集成 Pandas 与 Matplotlib 的数据分析沙盒，支持动态读取 CSV 并生成统计图表。

## 🚀 快速开始 (Quick Start with Docker)

本项目已完成全栈容器化，底层依赖（包括科学计算库与运行环境）已完全封装。

### 1. 前置准备
在首次运行前，请确保生成本地临床测试数据（Agent 数据分析沙盒的运行底座）：
```bash
python generate_mock_data.py
```

### 2. 构建与运行
请确保本地已安装运行 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，在终端执行：
```bash
# 构建 Docker 镜像
docker build -t pharma-agent:v3.1 .

# 启动服务并映射至本地 8000 端口
docker run -p 8000:8000 pharma-agent:v3.1
```

### 3. API 接口调试
容器启动成功后，访问交互式 API 文档 (Swagger UI)：
👉 **`http://127.0.0.1:8000/docs`**

Agent 自动生成的数据分析图表可通过静态路由直接访问，例如：
👉 **`http://127.0.0.1:8000/clinical_data/api_test_chart.png`**

*(注：以下为 V2.0 阶段 FastAPI 基础接口运行演示视频)*

[https://github.com/user-attachments/assets/211631dc-3af6-4e0e-b46e-2273bac580b8](https://github.com/user-attachments/assets/211631dc-3af6-4e0e-b46e-2273bac580b8)

---

## 🔌 API 调用规范 (API Reference)

系统提供标准的 RESTful API 接口，引入了 `session_id` 以支持多用户并发隔离。

**Endpoint:** `POST /agent/chat`

**请求示例 (触发三大引擎的终极测试指令):**

```json
{
  "question": "1. 查阅本地知识库，简述MARIPOSA试验中TP53的影响。2. 读取 clinical_data/lung_cancer_mock_data.csv，计算携带TP53共突变与未携带者的平均OS差异，用 matplotlib 画出图表并保存为 clinical_data/api_test_chart.png (请使用英文标注)。3. 去 PubMed 检索最新全网文献，与本地计算结果对比总结。",
  "session_id": "api_integration_test_01"
}
```

**响应示例:**

```json
{
  "answer": "## 综合研究报告...\n\n### 1. 本地文献分析\n根据 MARIPOSA 试验...\n\n### 2. 数据验证与可视化\n计算得出 TP53 共突变患者平均 OS 劣势为 3.1 个月。可视化图表已保存至 clinical_data/api_test_chart.png...\n\n### 3. 全球最新进展 (PubMed)\n根据检索到的最新文献 (PMID: 41510380)...",
  "session_id": "api_integration_test_01"
}
```

---

## 🖥️ 附加测试工具：Gradio UI (Legacy V1.0)

本项目保留了早期 V1.0 版本的交互式 Web UI，提供单纯的 RAG 打字机流式输出体验（不包含 Agent 路由与代码沙盒模块）。该模式脱离 Docker 运行，主要用于算法基础测试。

[https://github.com/user-attachments/assets/50fdceb4-10a0-49ac-9d7a-a1b6283ac31d](https://github.com/user-attachments/assets/50fdceb4-10a0-49ac-9d7a-a1b6283ac31d)

### 运行方式
```bash
python app.py
```
启动后访问 `http://127.0.0.1:7860` 即可进入图形化问答界面。