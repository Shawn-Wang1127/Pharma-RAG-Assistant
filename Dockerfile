# 使用官方 Python 3.10 轻量级镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装 (利用 Docker 缓存机制)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件到工作目录
COPY . .

# 关键基建：预先创建系统运行必须的挂载目录
# clinical_data: 供代码沙盒生成图表及 API 静态路由使用
# data / chroma_db: 供 RAG 向量数据库使用
RUN mkdir -p clinical_data data chroma_db chroma_db_bge_m3

# 暴露 FastAPI 运行端口
EXPOSE 8000

# 启动 Uvicorn 服务器，绑定 0.0.0.0 以允许宿主机访问
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]