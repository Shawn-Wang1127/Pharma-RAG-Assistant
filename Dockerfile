# 1. Base Image: Use the official lightweight Python 3.10 image
FROM python:3.10-slim

# 2. Working Directory: Set the working directory inside the container
WORKDIR /app

# 3. Cache Optimization: Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies using Aliyun mirror for faster build times
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. Application Code: Copy the entire project and vector database into the container
COPY . .

# 5. Expose Port: Expose the port that the ASGI server will listen on
EXPOSE 8000

# 6. Entry Point: Define the default command to start the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]