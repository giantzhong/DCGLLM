DATASET = "MIMIC4"
DATA_DICTIONARY = "data/"
DEFAULT_TARGET = "dm"  # comorbidity task: dm or cvd

# original data
MIMIC3_DATA_PATH = DATA_DICTIONARY + "MIMIC3/HTN2DM/sample_dataset_mimiciii_htn2dm_multifea.pkl"
# MIMIC4_DATA_PATH = DATA_DICTIONARY + "MIMIC4/sample_dataset_mimiciv_mortality_multifea.pkl"
MIMIC4_DATA_PATH = DATA_DICTIONARY + "MIMIC4/HTN2DM/sample_dataset_mimiciv_htn2dm_multifea.pkl"
# HuaDong (default naming; can be overridden from CLI)
HUADONG_DATA_PATH = DATA_DICTIONARY + "HuaDong/HTN2DM/sample_dataset_huadong_htn2dm_multifea.pkl"

# generated data
PATIENT_DATASET_PATH = "diagnoses.json"
ID_TO_ICD_MAP_PATH = "id_to_icd.json"
ICD_TO_ID_MAP_PATH = "icd_to_id.json"
CODE_TO_ICD_MAP_PATH = "code_to_icd_map.json"
CODE_TO_ID_MAP_PATH = "code_to_id.json"
EHR_DATA_PATH = "ehr_data.json"
TRUNCED_EHR_DATA_PATH = "ehr_data_trunced.json"
GRAPH_NODE_MASK_PATH = "node_masks.json"
CFIPF_DICT_PATH = "cfipf_dict.pkl"
CFIPF_MATRIX_PATH = "cfipf_matrix.npy"

class DataConfig:
    def __init__(self, dataset: str = DATASET, target: str = DEFAULT_TARGET):
        self.dataset = dataset

        # Normalize target -> "DM" / "CVD" / "CKD"
        t = (target or "").strip().upper()
        if t in ("DM", "CVD", "CKD"):
            target_up = t
        elif t in ("dm", "cvd", "ckd"):
            target_up = t.upper()
        else:
            raise ValueError(f"Unknown target={target!r}. Use 'dm', 'cvd', or 'ckd'.")

        self.target = target_up
        self.task_dir = f"HTN2{self.target}"
        self.output_path = DATA_DICTIONARY + dataset + "/" + self.task_dir

        if dataset == "MIMIC3":
            self.dataset_path = (
                DATA_DICTIONARY
                + "MIMIC3/"
                + self.task_dir
                + f"/sample_dataset_mimiciii_htn2{self.target.lower()}_multifea.pkl"
            )
        elif dataset == "MIMIC4":
            self.dataset_path = (
                DATA_DICTIONARY
                + "MIMIC4/"
                + self.task_dir
                + f"/sample_dataset_mimiciv_htn2{self.target.lower()}_multifea.pkl"
            )
        elif dataset == "HuaDong":
            self.dataset_path = (
                DATA_DICTIONARY
                + "HuaDong/"
                + self.task_dir
                + f"/sample_dataset_huadong_htn2{self.target.lower()}_multifea.pkl"
            )
        else:
            assert False and "Dataset not support"

        base = self.output_path + "/"
        self.patient_dataset_path = base + PATIENT_DATASET_PATH
        self.id_to_icd_map_path = base + ID_TO_ICD_MAP_PATH
        self.icd_to_id_map_path = base + ICD_TO_ID_MAP_PATH
        self.code_to_icd_map_path = base + CODE_TO_ICD_MAP_PATH
        self.code_to_id_map_path = base + CODE_TO_ID_MAP_PATH
        self.ehr_data_path = base + EHR_DATA_PATH
        self.trunced_ehr_data_path = base + TRUNCED_EHR_DATA_PATH
        self.node_masks_path = base + GRAPH_NODE_MASK_PATH
        self.cfipf_dict_path = base + CFIPF_DICT_PATH
        self.cfipf_matrix_path = base + CFIPF_MATRIX_PATH

        # Backward-compatible helpers (defaults point to HTN2DM)
        self.mimic3_dataset_path = MIMIC3_DATA_PATH
        self.mimic4_dataset_path = MIMIC4_DATA_PATH
        self.huadong_dataset_path = HUADONG_DATA_PATH
