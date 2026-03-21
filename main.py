import os
from core import BioAssistant

def run_project():
    # 1. 初始化
    assistant = BioAssistant()
    
    # 2. 自动检查数据库状态 (对应 50+ 文件量级的逻辑)
    db_path = "./chroma_db_bge_m3"
    data_folder = "data"
    
    if not os.path.exists(db_path):
        print("首次运行或模型更换，正在构建医药知识库索引...")
        assistant.build_vector_db(data_folder)
    else:
        # 获取 data 文件夹文件数量用于简历成果描述
        file_count = len([f for f in os.listdir(data_folder) if f.endswith('.pdf')])
        print(f"已加载本地知识库，当前索引涵盖 {file_count} 份核心医药文献。")

    # 3. 开启测试
    # 测试问题示例（可更换为你新下载的 PubMed 文献内容相关问题）
    query = "根据 MARIPOSA 试验的相关文献,Amivantamab 联合 lazertinib 相比于奥希替尼单药，在晚期 EGFR 突变非小细胞肺癌中的疗效（如无进展生存期 PFS 或总生存期 OS)表现如何？"
    
    answer, sources = assistant.rag_chat(query)
    
    # 4. 结果展示
    print("\n" + "="*50)
    print(f"【用户提问】: {query}")
    print("-" * 50)
    print(f"【AI 专业回复】:\n{answer}")
    print("-" * 50)
    print("【参考来源清单】:")
    # 来源去重显示
    seen_sources = set()
    for doc in sources:
        source_name = doc.metadata.get('source', 'Unknown')
        if source_name not in seen_sources:
            print(f" - {source_name}")
            seen_sources.add(source_name)
    print("="*50)

if __name__ == "__main__":
    run_project()