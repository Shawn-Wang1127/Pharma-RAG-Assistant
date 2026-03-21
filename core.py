import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

class BioAssistant:
    def __init__(self):
        # 1. 初始化对话模型 (DeepSeek-V3)
        self.llm = ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
            max_tokens=2048
        )
        
        # 2. 优化切片策略：适应专业长难句，增加重叠度确保语义连续
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,    
            chunk_overlap=300,  
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # 3. 初始化 BGE-M3 向量模型
        print("正在初始化 BGE-M3 向量模型 (CPU 模式)...")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )   
        
        # 4. 指定向量库存储路径 (BGE-M3 专用库)
        self.persist_directory = "./chroma_db_bge_m3"
        self.vector_db = None

    def build_vector_db(self, data_folder: str):
        """批量扫描 PDF -> 切片 -> 向量化 -> 存入数据库"""
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
        if not pdf_files:
            print("错误：data 文件夹中未发现 PDF 文件。")
            return
            
        all_chunks = []
        for file_name in pdf_files:
            print(f"正在加载: {file_name}")
            loader = PyPDFLoader(os.path.join(data_folder, file_name))
            all_chunks.extend(self.text_splitter.split_documents(loader.load()))
        
        print(f"正在执行向量化并持久化 {len(all_chunks)} 个片段...")
        self.vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"向量库构建成功：{self.persist_directory}")

    def query_knowledge(self, english_query: str):
        """使用翻译后的英文关键词检索最相关的 10 个片段"""
        if not os.path.exists(self.persist_directory):
            print("未发现本地数据库，请先执行 build_vector_db。")
            return []
            
        if self.vector_db is None:
            self.vector_db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
        
        # 检索深度设为 10，保证 50+ 文件时的召回率
        docs = self.vector_db.max_marginal_relevance_search(
            english_query, 
            k=10,            # 最终塞给 AI 的片段数量
            fetch_k=30,      # 初始候选池大小（先捞出30个相关的，再从中挑10个差异大的）
            lambda_mult=0.5  # 0.5表示相关性和多样性各占一半权重（0更看重多样性，1更看重相关性）
        )
        return docs
    
    def rag_chat(self, question: str):
        """全方位医药智能检索逻辑：翻译 -> 检索 -> 增强生成"""
        # A. 查询优化：将中文问题转化为专业英文关键词
        print(f"--- 正在翻译查询以优化检索 ---")
        translation_prompt = (
            "You are a medical expert. Translate the following Chinese question into "
            "professional English search keywords. Output ONLY the English keywords:\n"
            f"{question}"
        )
        english_query = self.llm.invoke(translation_prompt).content.strip()
        print(f"优化后的搜索词: {english_query}")

        # B. 执行检索
        print(f"--- 正在从多源数据库检索参考资料 ---")
        context_docs = self.query_knowledge(english_query)
        if not context_docs:
            return "未能检索到相关资料。", []

        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # C. 构建增强型系统 Prompt
        prompt_template = ChatPromptTemplate.from_template("""
        你是一位精通生物信息学和临床肿瘤学的资深助手。请基于提供的【参考资料】回答问题。

        ### 回答准则：
        1. **证据优先**：优先引用指南(Guideline)和药物标签(Label)中的数据。
        2. **深度总结**：如果资料未直接给出结论，请根据文中描述的耐药机制、临床表现进行专业归纳。
        3. **溯源规范**：请用中文回答，条理清晰，并在回答中提及参考的来源。

        ### 参考资料库:
        {context}
        
        ### 用户问题: 
        {question}
        
        请提供专业的深度解析：
        """)
        
        # D. 最终生成
        print(f"--- 正在咨询 AI 助手 ---")
        chain = prompt_template | self.llm
        response = chain.invoke({"context": context_text, "question": question})
        
        return response.content, context_docs