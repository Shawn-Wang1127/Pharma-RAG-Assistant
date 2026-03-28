import os
import pandas as pd
import numpy as np

# ==========================================
# 配置参数与目录初始化
# ==========================================
OUTPUT_DIR = "clinical_data"
FILE_NAME = "lung_cancer_mock_data.csv"
NUM_PATIENTS = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)  # 固定随机种子，确保每次生成的数据一致，方便复现

print("[SYSTEM] Starting mock clinical data generation...")

# ==========================================
# 模拟核心临床变量
# ==========================================
# 1. 基础人口学特征
patient_ids = [f"PT-{str(i).zfill(4)}" for i in range(1, NUM_PATIENTS + 1)]
ages = np.random.normal(62, 8, NUM_PATIENTS).astype(int)
genders = np.random.choice(['Male', 'Female'], NUM_PATIENTS)
smoking_history = np.random.choice(['Never', 'Former', 'Current'], NUM_PATIENTS, p=[0.5, 0.3, 0.2])

# 2. 基因突变类型 (重点引入 TP53 高危共突变)
mutations = np.random.choice(['Ex19del', 'L858R', 'Ex19del+TP53', 'L858R+TP53'], NUM_PATIENTS, p=[0.35, 0.35, 0.15, 0.15])

# 3. 治疗方案 (对照组 vs 联合组)
treatments = np.random.choice(['Osimertinib_Mono', 'Amivantamab_Lazertinib_Combo'], NUM_PATIENTS)

# ==========================================
# 模拟生存期数据 (PFS 与 OS)
# ==========================================
pfs_months = []
os_months = []
status = [] # 1 代表死亡 (Event)，0 代表存活/失访 (Censored)

for i in range(NUM_PATIENTS):
    # 根据我们在本地知识库中查到的医学逻辑设定基线
    base_pfs = 16.6 if treatments[i] == 'Osimertinib_Mono' else 23.7
    
    # 设定高危突变的负向权重
    if 'TP53' in mutations[i]:
        base_pfs -= 5.0
        
    # 加入高斯噪声模拟个体差异
    patient_pfs = max(2.0, np.random.normal(base_pfs, 4.0))
    # OS 通常大于 PFS
    patient_os = patient_pfs + max(5.0, np.random.normal(15.0, 6.0))
    
    pfs_months.append(round(patient_pfs, 1))
    os_months.append(round(patient_os, 1))
    
    # 根据 OS 长度动态生成生存状态 (存活时间越短，记录为死亡的概率越高)
    stat = np.random.choice([1, 0], p=[0.7, 0.3]) if patient_os < 35 else np.random.choice([1, 0], p=[0.3, 0.7])
    status.append(stat)

# ==========================================
# 构建 DataFrame 并导出
# ==========================================
df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Age': ages,
    'Gender': genders,
    'Smoking_History': smoking_history,
    'Mutation_Type': mutations,
    'Treatment_Arm': treatments,
    'PFS_Months': pfs_months,
    'OS_Months': os_months,
    'Survival_Status': status
})

output_path = os.path.join(OUTPUT_DIR, FILE_NAME)
df.to_csv(output_path, index=False)

print(f"[SYSTEM] Successfully generated {NUM_PATIENTS} patient records.")
print(f"[SYSTEM] Data saved to: {output_path}")
print(f"[SYSTEM] Data preview:")
print(df.head())