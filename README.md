# Pharma-RAG Assistant


基于本地知识库的医药文献检索与增强生成 (RAG) 系统。本项目专为高严谨性的生物医药领域文献设计，整合了本地向量检索与 LLM 生成能力，通过 FastAPI 提供标准后端接口，并支持 Docker 容器化一键部署。


## 💡 核心架构 (Architecture)

* **LLM 引擎**: DeepSeek-V3 (兼容 OpenAI API 规范)
* **Embedding 模型**: BAAI/bge-m3 (本地 CPU 运行，彻底保障医学数据隐私)
* **向量数据库**: ChromaDB
* **后端服务框架**: FastAPI + Uvicorn
* **数据校验**: Pydantic
* **核心检索策略**: 基于最大边际相关性 (MMR) 算法，平衡信源的准确性与多样性。

## 🚀 快速开始 (Quick Start with Docker)

本项目已完成全栈容器化，底层依赖与运行环境已完全封装，宿主机无需配置 Python 环境即可运行。

### 1. 前置环境
- 请确保本地已安装并运行 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 或 Docker Engine。

### 2. 构建与运行
在项目根目录下，打开终端执行以下命令：

```bash
# 构建 Docker 镜像 (初次构建可能需要几分钟下载环境)
docker build -t pharma-rag:v2 .

# 启动服务并映射至本地 8000 端口
docker run -p 8000:8000 pharma-rag:v2
```

### 3. API 接口调试
容器启动成功后，请在浏览器中访问交互式 API 文档 (Swagger UI)：
👉 **`http://127.0.0.1:8000/docs`**


https://github.com/user-attachments/assets/211631dc-3af6-4e0e-b46e-2273bac580b8


---

## 🔌 API 调用规范 (API Reference)

系统提供标准的 RESTful API 接口，可轻松接入企业级前端 (Vue/React) 或其他微服务中。

**Endpoint:** `POST /chat`

**请求示例 (Request Payload):**

```json
{
  "question": "SCLC转化的机制是什么？"
}
```

**响应示例 (Response):**

```json
{
  "answer": "根据提供的参考资料，EGFR突变型非小细胞肺癌向小细胞肺癌转化的机制涉及RB1和TP53的双重失活、谱系可塑性以及神经内分泌转录重编程...",
  "sources": [
    "data/Evolutionary and transcriptomic mechanisms.pdf",
    "data/Advances in molecular pathology.pdf"
  ]
}
```

---

## 🖥️ 附加测试工具：Gradio UI

本项目保留了 V1.0 版本的交互式 Web UI (基于 Gradio)，提供打字机流式输出的直观体验。该模式脱离 Docker 运行，主要用于本地开发与算法效果测试。

https://github.com/user-attachments/assets/50fdceb4-10a0-49ac-9d7a-a1b6283ac31d

### 运行方式
需在本地配置 Python 3.10+ 并安装 `requirements.txt` 后，运行：

```bash
python app.py
```
启动后访问 `http://127.0.0.1:7860` 即可进入图形化问答界面。