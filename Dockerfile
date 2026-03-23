# 1. 声明基础镜像：使用官方的 Python 3.10 轻量级镜像
FROM python:3.10-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 复制依赖清单并安装
# 先复制 requirements.txt 可以利用 Docker 的缓存机制，加速后续构建
COPY requirements.txt .
# 使用阿里云镜像源加速安装，并安装所需的系统基础库
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. 把你当前目录下的所有代码和数据库拷贝进容器的 /app 目录
COPY . .

# 5. 暴露容器的 8000 端口
EXPOSE 8000

# 6. 定义容器启动时执行的默认命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]