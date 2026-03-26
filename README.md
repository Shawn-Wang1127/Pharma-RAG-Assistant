# Pharma-Agent (原 Pharma-RAG Assistant)

基于 LangGraph 状态机与本地向量检索构建的医疗医药智能体（Agent）后端系统。本项目从基础的 RAG 架构升级而来，具备自主决策、工具调用、多租户并发记忆跟踪以及完整的 RESTful API 接入能力。

## 核心架构 (Architecture)

* **Agent 编排框架**: LangGraph (支持 ReAct 路由、状态机管理与 MemorySaver 持久化记忆)
* **LLM 引擎**: DeepSeek-V3 (兼容 OpenAI API 规范，支持 Tool Calling)
* **Embedding 模型**: BAAI/bge-m3 (本地 CPU 运行，彻底保障医学数据隐私)
* **向量数据库**: langchain-chroma
* **后端服务框架**: FastAPI + Uvicorn + Pydantic
* **工具集 (Tools)**:
  * `search_medical_literature`: 封装了基于 MMR 算法的本地 RAG 搜索引擎。
  * `calculate_drug_half_life`: 药代动力学参数检索工具。

## 快速开始 (Quick Start with Docker)

本项目已完成全栈容器化，底层依赖与运行环境已完全封装，宿主机无需配置 Python 环境即可运行。

### 1. 前置环境
- 请确保本地已安装并运行 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 或 Docker Engine。

### 2. 构建与运行
在项目根目录下，打开终端执行以下命令：

```bash
# 构建 Docker 镜像
docker build -t pharma-agent:v3 .

# 启动服务并映射至本地 8000 端口
docker run -p 8000:8000 pharma-agent:v3
```

### 3. API 接口调试
容器启动成功后，请在浏览器中访问交互式 API 文档 (Swagger UI)：
👉 **`http://127.0.0.1:8000/docs`**

*(注：以下为 V2.0 阶段 FastAPI 基础接口运行演示视频)*

[https://github.com/user-attachments/assets/211631dc-3af6-4e0e-b46e-2273bac580b8](https://github.com/user-attachments/assets/211631dc-3af6-4e0e-b46e-2273bac580b8)

---

## 🔌 API 调用规范 (API Reference)

系统提供标准的 RESTful API 接口。V3.0 版本引入了 `session_id` 以支持多用户的并发上下文隔离记忆。

**Endpoint:** `POST /agent/chat`

**请求示例 (Request Payload):**

```json
{
  "question": "根据 MARIPOSA 试验，Amivantamab 联合 lazertinib 的疗效如何？",
  "session_id": "user_12345"
}
```

**响应示例 (Response):**

```json
{
  "answer": "根据检索到的文献，在 MARIPOSA 试验中，Amivantamab 联合 Lazertinib 一线治疗 EGFR 突变晚期非小细胞肺癌，相比奥希替尼单药显示出显著的 PFS 优势...\n\n文献来源:\n- data/Amivantamab plus lazertinib vs. osimertinib.pdf",
  "session_id": "user_12345"
}
```

---

## 🖥️ 附加测试工具：Gradio UI (Legacy V1.0)

本项目保留了早期版本的交互式 Web UI (基于 Gradio)，提供单纯的 RAG 打字机流式输出直观体验（不包含 Agent 路由与记忆模块）。该模式脱离 Docker 运行，主要用于本地 RAG 算法效果的基础测试。

[https://github.com/user-attachments/assets/50fdceb4-10a0-49ac-9d7a-a1b6283ac31d](https://github.com/user-attachments/assets/50fdceb4-10a0-49ac-9d7a-a1b6283ac31d)

### 运行方式
需在本地配置 Python 3.10+ 并安装 `requirements.txt` 后，运行：

```bash
python app.py
```
启动后访问 `http://127.0.0.1:7860` 即可进入图形化问答界面。