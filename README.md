# Pharma-RAG: 医药文献智能辅助决策系统

基于多模态医学文献（临床指南、PubMed 顶刊、FDA Label）构建的高精度本地 RAG（检索增强生成）问答系统。



https://github.com/user-attachments/assets/50fdceb4-10a0-49ac-9d7a-a1b6283ac31d



## 🌟 项目亮点
- **处理管线**：支持解析双栏排版与复杂表格的生物医药 PDF，定制化 `RecursiveCharacterTextSplitter` 切片策略，有效保留上下文。
- **高精度跨语言检索**：集成 `BAAI/bge-m3` 多语言稠密检索模型，突破中英医学术语的向量映射壁垒。
- **检索优化 (MMR)**：引入 Maximal Marginal Relevance 算法进行检索后重排，平衡召回率与结果多样性，缓解答案同质化。
- **Zero-Hallucination**：采用 Query Translation 代理机制，强制大模型输出 100% 具备文献溯源的推理性结论。

## 🛠️ 技术栈
- **框架**: LangChain, Gradio
- **大模型**: DeepSeek-V3 (经由 OpenAI API 协议)
- **向量数据库**: ChromaDB
- **Embedding模型**: BGE-M3 (HuggingFace)

## 🚀 快速开始

### 1. 克隆与安装
```bash
git clone [https://github.com/你的用户名/Pharma-RAG.git](https://github.com/你的用户名/Pharma-RAG.git)
cd Pharma-RAG
pip install -r requirements.txt
