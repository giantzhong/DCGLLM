BATCH_SIZE = 32
HIDDEN_CHANNELS = 1024
GRAPH_VECTOR_SIZE = 8

DATASET = "MIMIC3" # "MIMIC4"
DATASET_DICTIONARY = "data/"

# LLM_MODEL_NAME = "HuatuoGPT_II"
LLM_MODEL_NAME = "BioMistral-7B"
# generated data
GRAPH_DATASET_PATH = DATASET_DICTIONARY + DATASET + f"/graph_{LLM_MODEL_NAME}.pkl"
TRUNCED_DATA_PATH = DATASET_DICTIONARY + DATASET + "/" + "diagnoses_trunc.json"

# LLM config
# 每个HuatuoGPT-II模型约需26-28GB显存（FP16）
# RTX 4090: 24GB显存，建议每个GPU加载1个模型

# 模型数量和路径
LLM_MODEL_NUM = 4  # 每个GPU加载2个模型 → 4个模型实例 → 更高GPU利用率
# LLM_MODEL_PATH = "/home/chenqianzhong24/model/HuatuoGPT_II"
LLM_MODEL_PATH = f"/home/chenqianzhong24/model/{LLM_MODEL_NAME}"
LLM_MAX_CONTEXT_LENGTH = 32768

# GPU分配策略
# 选项1: 手动指定GPU (推荐，更可控)
LLM_GPU_IDS = [1,2,3,5]  # GPU 0加载模型0,2；GPU 1加载模型1,3（轮询分配）

# 选项2: 自动分配 (不推荐多GPU场景)
# LLM_GPU_IDS = None  # 使用device_map="auto"

# 性能说明:
# - 3个模型 → 3倍并发 → 速度提升约3倍
# - 每个4090显存: 24GB，模型需要: ~26GB (会有一点紧张)
# - 如果显存不足，可以考虑量化或减少模型数量

PATIENT_VISIT_PROMPT_TEMPLATE = """\
Patient Visit History

The patient has multiple hospital visits.
Each visit lists: (a) diagnosed diseases, and (b) the number of days since the previous visit.

{VISIT_LIST}
"""

PATIENT_TASK_PROMPT_TEMPLATE = """\
Task Instruction

You are a medical reasoning assistant.

Based on the Patient Visit History above, determine whether {DISEASE_A} is clinically related to {DISEASE_B} for this specific patient.

Your reasoning should rely only on:
1. Co-occurrence of diseases within the same visit,
2. Recurrence patterns across visits, and
3. Temporal intervals between visits.

Answer the following question with "Yes" or "No":
Is {DISEASE_A} clinically related to {DISEASE_B} for this patient? Yes or No.
"""

PERPLEXITY_TEXT_TEMPLATE = \
    "{} is related to "

LLM_MAKE_SCORE= ''

from data_preprocess.data_config import DataConfig
dataConfig = DataConfig(DATASET)

PATIENT_DATASET_PATH = dataConfig.patient_dataset_path 
ICD_TO_ID_MAP_PATH = dataConfig.icd_to_id_map_path
EHR_DATA_PATH = dataConfig.ehr_data_path 

class Config:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.hidden_channels = HIDDEN_CHANNELS
        self.graph_vector_size = GRAPH_VECTOR_SIZE
        self.graph_dataset_path = GRAPH_DATASET_PATH
        # self.cfipf_matrix_path = dataConfig.node_feature
        self.node_id_mask_path = dataConfig.node_masks_path

        self.llm_model_num = LLM_MODEL_NUM
        self.llm_model_path = LLM_MODEL_PATH
        self.llm_max_context_length = LLM_MAX_CONTEXT_LENGTH
        self.patient_visit_prompt_template = PATIENT_VISIT_PROMPT_TEMPLATE
        self.patient_task_prompt_template = PATIENT_TASK_PROMPT_TEMPLATE
        self.perplexity_text_template = PERPLEXITY_TEXT_TEMPLATE
