"""
Script to prepare MIMIC-IV dataset using PyHealth
This script generates the sample_dataset_mimiciv_mortality_multifea.pkl file
With custom task function to include delta_days (time intervals between visits)
"""

import os
from datetime import datetime
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import mortality_prediction_mimic4_fn
from pyhealth.utils import save_pickle

from comorbidity_icdcode import HTN_REGEX, get_target_regex, normalize_target

# Configuration
MIMIC4_ROOT = "/home/chenqianzhong24/dataset/MIMIC-IV/hosp"
OUTPUT_DIR = "/home/chenqianzhong24/project/DCGLLM/data/MIMIC4"
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_dataset_mimiciv_mortality_multifea.pkl")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_dataset_mimiciv_cormobidity_multifea.pkl")

# Comorbidity target: "DM" or "CVD" (will be overwritten by CLI if provided)
TARGET_DISEASE = "DM"


def custom_mortality_prediction_mimic4_fn(patient):
    """
    Custom task function for MIMIC-IV FINAL-VISIT mortality prediction
    
    DCGLLM Task Definition:
    - Input (X): All historical visits (visit 1 to T-1) for a patient
    - Label (Y): Mortality status at the final visit (visit T)
    - Result: ONE sample per patient
    
    This implements the "final-visit mortality prediction" task where we use
    all historical visits to predict mortality at the last visit.
    
    Returns:
        List containing at most ONE sample with:
        - visit_id, patient_id
        - conditions, procedures, drugs (lists of lists, one per visit)
        - delta_days: time intervals between consecutive visits
        - label: mortality outcome at final visit
    """
    samples = []
    
    # Get number of visits (Patient object supports len() but not negative indexing)
    num_visits = len(patient)
    
    # Need at least 2 visits: 1 for history, 1 for prediction
    if num_visits < 2:
        return samples
    
    # Get the final (last) visit for the label using positive index
    final_visit = patient[num_visits - 1]
    
    # Determine mortality label from final visit's discharge status
    if final_visit.discharge_status not in [0, 1]:
        mortality_label = 0
    else:
        mortality_label = int(final_visit.discharge_status)
    
    # Initialize lists to store data from all historical visits
    all_conditions = []
    all_procedures = []
    all_drugs = []
    all_delta_days = []
    previous_valid_visit = None
    
    # Process each historical visit (all visits BEFORE the final visit)
    for i in range(num_visits - 1):
        visit = patient[i]
        # Extract medical codes
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        
        # Exclude visits without all three types of codes
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        
        # Calculate delta_days (time since previous valid visit)
        delta_days_value = 0
        
        if previous_valid_visit is not None:
            # Calculate time difference from previous valid visit
            try:
                if hasattr(visit, 'encounter_time') and hasattr(previous_valid_visit, 'encounter_time'):
                    current_time = visit.encounter_time
                    previous_time = previous_valid_visit.encounter_time
                    
                    if current_time and previous_time:
                        # Handle different time formats
                        if isinstance(current_time, str):
                            current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                        if isinstance(previous_time, str):
                            previous_time = datetime.fromisoformat(previous_time.replace('Z', '+00:00'))
                        
                        delta_days_value = max(0, (current_time - previous_time).days)
                    else:
                        delta_days_value = 1
                else:
                    delta_days_value = 1
            except Exception as e:
                delta_days_value = 1
        
        # Append this visit's data
        all_conditions.append(conditions)
        all_procedures.append(procedures)
        all_drugs.append(drugs)
        all_delta_days.append([delta_days_value])  # Keep [[value]] format
        
        # Update previous valid visit
        previous_valid_visit = visit
    
    # Only create a sample if we have at least one valid historical visit
    if len(all_conditions) > 0:
        samples.append({
            "visit_id": final_visit.visit_id,  # ID of the final visit we're predicting
            "patient_id": patient.patient_id,
            "conditions": all_conditions,  # List of lists, one per historical visit
            "procedures": all_procedures,  # List of lists, one per historical visit
            "drugs": all_drugs,            # List of lists, one per historical visit
            "delta_days": all_delta_days,  # List of [value], one per historical visit
            "label": mortality_label,      # Mortality at final visit
        })
    
    return samples

def prepare_mimic4_dataset():
    """
    Prepare MIMIC-IV dataset for mortality prediction with multiple features
    """
    
    print("=" * 80)
    print("MIMIC-IV Data Preparation using PyHealth")
    print("=" * 80)
    
    # Step 1: Load MIMIC-IV base dataset
    print("\n[Step 1/4] Loading MIMIC-IV base dataset...")
    print(f"Reading from: {MIMIC4_ROOT}")
    
    try:
        base_dataset = MIMIC4Dataset(
            root=MIMIC4_ROOT,
            tables=[
                "diagnoses_icd",
                "procedures_icd", 
                "prescriptions",
                # "labevents"
            ],
            code_mapping={"ICD9CM": "CCSCM", "ICD10CM": "CCSCM"},
            dev=False,  # Set to True for quick testing with small subset
            refresh_cache=False  # Set to True to force reprocess
        )
        print(f"✓ Base dataset loaded successfully!")
        print(f"  - Number of patients: {len(base_dataset.patients)}")
        print(f"  - Available tables: {base_dataset.tables}")
        
    except Exception as e:
        print(f"✗ Error loading base dataset: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Verify MIMIC-IV data path is correct")
        print("  2. Ensure all required CSV files are present")
        print("  3. Try setting dev=True for testing with small subset")
        raise
    
    # Step 2: Create mortality prediction task with custom function
    print("\n[Step 2/4] Creating mortality prediction task with delta_days...")
    
    try:
        sample_dataset = base_dataset.set_task(
            task_fn=custom_mortality_prediction_mimic4_fn,
            task_name="mortality"
        )
        
        print(f"✓ Task created successfully!")
        print(f"  - Number of samples: {len(sample_dataset)}")
        
        # Print sample information
        if len(sample_dataset) > 0:
            sample = sample_dataset[0]
            print(f"\n  Sample structure:")
            print(f"    - Available keys: {sample.keys()}")
            if 'conditions' in sample:
                print(f"    - Conditions (diagnoses): {len(sample['conditions'])} visits")
            if 'procedures' in sample:
                print(f"    - Procedures: {len(sample['procedures'])} visits")
            if 'drugs' in sample:
                print(f"    - Drugs: {len(sample['drugs'])} visits")
            if 'delta_days' in sample:
                print(f"    - Delta days: {sample['delta_days']}")
            if 'label' in sample:
                print(f"    - Label: {sample['label']}")
                
    except Exception as e:
        print(f"✗ Error creating task: {e}")
        raise
    
    # Step 3: Save to pickle file
    print("\n[Step 3/4] Saving dataset to pickle file...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save the dataset
        save_pickle(sample_dataset, OUTPUT_FILE)
        
        print(f"✓ Dataset saved successfully!")
        print(f"  - Output path: {OUTPUT_FILE}")
        print(f"  - File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        raise
    
    # Step 4: Verify the saved file
    print("\n[Step 4/4] Verifying saved file...")
    
    try:
        from pyhealth.utils import load_pickle
        
        loaded_dataset = load_pickle(OUTPUT_FILE)
        print(f"✓ Verification successful!")
        print(f"  - Loaded {len(loaded_dataset)} samples")
        
    except Exception as e:
        print(f"✗ Error verifying file: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("✓ Data preparation completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: python data_preprocess/utils.py")
    print("  2. This will generate all intermediate files needed for the model")
    print("=" * 80)
    
    return sample_dataset


    """
    Custom task function for MIMIC-IV HTN->DM comorbidity prediction (ONE sample per patient)

    任务定义（与前文一致的多访视输入 X、单标签 Y）：
    - 目标：预测高血压（HTN）患者未来是否会发展为糖尿病（DM）
    - 样本（每位患者最多 1 条）：
        X = 截止到“预测截断点”之前的所有历史有效访视（条件/手术/用药 + delta_days）
        Y = 是否在 HTN 之后（且非同次/同时时间）发生首个 DM（1=会，0=不会）
    - 排除规则：
        1) 先被诊断为DM，再被诊断为HTN（DM_time < HTN_time）
        2) HTN与DM在同一次住院或同一时间（同visit_id或同encounter_time）
        3) 住院/访视次数 < 2（len(patient) < 2）

    返回字段与前一个任务保持一致：
        - visit_id: 对于case=首个DM的visit_id；对于control=最后一次有效历史visit_id
        - patient_id
        - conditions/procedures/drugs: [ [codes_of_visit1], [codes_of_visit2], ... ]
        - delta_days: [ [d1], [d2], ... ]  # 与你现有脚本保持 [[value]] 形式
        - label: 1/0
    """
    samples = []
    from pyhealth.medcode import CrossMap

    # 高血压 (HTN)
    icd_htn_list = [
        # ICD-9 CM
        "401", "402", "403", "404", "405",
        # ICD-10 CM
        "I10", "I11", "I12", "I13", "I15"
    ]

    # 糖尿病 (DM)
    icd_dm_list = [
        # ICD-9 CM
        "250",
        # ICD-10 CM
        "E08", "E09", "E10", "E11", "E12", "E13"
    ]

    map_icd9_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
    map_icd10_to_ccs = CrossMap.load("ICD10CM", "CCSCM")

    ccs_htn = set(map_icd9_to_ccs.map_batch(icd_htn_list) + map_icd10_to_ccs.map_batch(icd_htn_list))
    ccs_dm  = set(map_icd9_to_ccs.map_batch(icd_dm_list)  + map_icd10_to_ccs.map_batch(icd_dm_list))

    # 至少需要 2 次访视（历史+未来/监测窗口）
    num_visits = len(patient)
    if num_visits < 2:
        return samples

    # ---- ICD 规则（与你上面的实现保持一致）----
    import re
    htn_pat = re.compile(r'^(401|402|403|404|405|I10|I11|I12|I13|I15)', re.IGNORECASE)
    dm_pat  = re.compile(r'^(250|E0[8-9]|E1[0-3])',                   re.IGNORECASE)

    # 工具：判断某次访视是否含HTN/DM诊断
    def has_icd(visit, pat, table):
        codes = visit.get_code_list(table=table)  # e.g., "diagnoses_icd"
        for c in codes:
            if pat.match(str(c)):
                return True
        return False

    # 逐访视扫描，找到首个HTN与首个DM的索引、visit_id、时间
    first_htn_idx, first_dm_idx = None, None
    first_htn_visit_id, first_dm_visit_id = None, None
    first_htn_time, first_dm_time = None, None

    def get_time(v):
        # 与你现有 delta_days 计算保持一致：优先用 encounter_time
        t = getattr(v, "encounter_time", None)
        return t

    for i in range(num_visits):
        v = patient[i]
        # 判断 HTN
        if first_htn_idx is None and has_icd(v, htn_pat, "diagnoses_icd"):
            first_htn_idx = i
            first_htn_visit_id = v.visit_id
            first_htn_time = get_time(v)
        # 判断 DM
        if first_dm_idx is None and has_icd(v, dm_pat, "diagnoses_icd"):
            first_dm_idx = i
            first_dm_visit_id = v.visit_id
            first_dm_time = get_time(v)

    # 如果没有HTN，无法构造样本
    if first_htn_idx is None:
        return samples

    # 排除规则 #3：访视数 < 2
    if num_visits < 2:
        return samples

    # 排除规则 #1：先DM后HTN
    if first_dm_idx is not None and first_dm_idx < first_htn_idx:
        return samples

    # 排除规则 #2：同次/同时时间
    def same_time(t1, t2):
        if (t1 is None) or (t2 is None):
            return False
        # 字符串时间规范化
        from datetime import datetime
        if isinstance(t1, str):
            t1 = datetime.fromisoformat(t1.replace('Z', '+00:00'))
        if isinstance(t2, str):
            t2 = datetime.fromisoformat(t2.replace('Z', '+00:00'))
        return t1 == t2

    simultaneous = (
        first_dm_idx is not None and
        first_htn_idx is not None and (
            first_dm_visit_id == first_htn_visit_id or
            same_time(first_dm_time, first_htn_time)
        )
    )
    if simultaneous:
        return samples

    # -------- 构造 X（历史有效访视）与 label --------
    # 有效访视：需同时具备 诊断/手术/用药 三类代码（与你原脚本一致）
    # 历史范围：
    #   - case（HTN->DM）：历史 = 首个DM之前的所有有效访视
    #   - control（HTN且从未DM）：历史 = 所有有效访视
    is_case = first_dm_idx is not None and first_dm_idx > first_htn_idx

    # 历史上限（不含上限）：case 截止首个DM；control 截止最后一次
    hist_end = first_dm_idx if is_case else num_visits

    all_conditions, all_procedures, all_drugs, all_delta_days = [], [], [], []
    previous_valid_visit = None
    last_hist_valid_visit_id = None

    from datetime import datetime

    for i in range(hist_end):
        v = patient[i]
        conds = v.get_code_list(table="diagnoses_icd")
        procs = v.get_code_list(table="procedures_icd")
        meds  = v.get_code_list(table="prescriptions")

        # 必须三者都有（与你原脚本逻辑保持一致）
        if len(conds) * len(procs) * len(meds) == 0:
            continue

        # delta_days
        delta_days_value = 0
        if previous_valid_visit is not None:
            try:
                t_cur = getattr(v, "encounter_time", None)
                t_prev = getattr(previous_valid_visit, "encounter_time", None)
                if t_cur and t_prev:
                    if isinstance(t_cur, str):
                        t_cur = datetime.fromisoformat(t_cur.replace('Z', '+00:00'))
                    if isinstance(t_prev, str):
                        t_prev = datetime.fromisoformat(t_prev.replace('Z', '+00:00'))
                    delta_days_value = max(0, (t_cur - t_prev).days)
                else:
                    delta_days_value = 1
            except Exception:
                delta_days_value = 1

        all_conditions.append(conds)
        all_procedures.append(procs)
        all_drugs.append(meds)
        all_delta_days.append([delta_days_value])
        last_hist_valid_visit_id = v.visit_id
        previous_valid_visit = v

    # 历史必须至少有 1 次有效访视
    if len(all_conditions) == 0:
        return samples

    # 标签与锚定 visit_id
    if is_case:
        label = 1
        anchor_visit_id = first_dm_visit_id  # 预测目标发生点
    else:
        label = 0
        anchor_visit_id = last_hist_valid_visit_id  # 控制组用最后一个历史有效访视

    samples.append({
        "visit_id": anchor_visit_id,
        "patient_id": patient.patient_id,
        "conditions": all_conditions,
        "procedures": all_procedures,
        "drugs": all_drugs,
        "delta_days": all_delta_days,
        "label": label,
    })
    return samples

def custom_cormobidity_prediction_mimic4_fnh(patient):
    """
    Custom task function for MIMIC-IV HTN->(DM/CVD) comorbidity prediction (ONE sample per patient)

    任务定义（与前文一致的多访视输入 X、单标签 Y）：
    - 目标：预测高血压（HTN）患者未来是否会发展为目标疾病（DM/CVD）
    - 样本（每位患者最多 1 条）：
        X = 截止到“预测截断点”之前的所有历史有效访视（条件/手术/用药 + delta_days）
        Y = 是否在 HTN 之后（且非同次/同时时间）发生首个 target（1=会，0=不会）
    - 排除规则：
        1) 先被诊断为target，再被诊断为HTN（target_time < HTN_time）
        2) HTN与target在同一次住院或同一时间（同visit_id或同encounter_time）
        3) 住院/访视次数 < 2（len(patient) < 2）

    返回字段与前一个任务保持一致：
        - visit_id: 对于case=首个target的visit_id；对于control=最后一次有效历史visit_id
        - patient_id
        - conditions/procedures/drugs: [ [codes_of_visit1], [codes_of_visit2], ... ]
        - delta_days: [ [d1], [d2], ... ]  # 与你现有脚本保持 [[value]] 形式
        - label: 1/0
    """
    samples = []

    # 至少需要 2 次访视（历史+未来/监测窗口）
    num_visits = len(patient)
    if num_visits < 2:
        return samples

    # ---- ICD 规则 ----
    import re
    htn_pat = re.compile(HTN_REGEX, re.IGNORECASE)
    target_pat = re.compile(get_target_regex(TARGET_DISEASE), re.IGNORECASE)

    # 工具：判断某次访视是否含HTN/target诊断
    def has_icd(visit, pat, table):
        codes = visit.get_code_list(table=table)  # e.g., "diagnoses_icd"
        for c in codes:
            # Normalize ICD code: upper-case and remove '.' (e.g., "401.1" -> "4011")
            icd_norm = str(c).upper().replace(".", "").strip()
            if pat.match(icd_norm):
                return True
        return False

    # 逐访视扫描，找到首个HTN与首个target的索引、visit_id、时间
    first_htn_idx, first_target_idx = None, None
    first_htn_visit_id, first_target_visit_id = None, None
    first_htn_time, first_target_time = None, None

    def get_time(v):
        # 与你现有 delta_days 计算保持一致：优先用 encounter_time
        t = getattr(v, "encounter_time", None)
        return t

    for i in range(num_visits):
        v = patient[i]
        # 判断 HTN
        if first_htn_idx is None and has_icd(v, htn_pat, "diagnoses_icd"):
            first_htn_idx = i
            first_htn_visit_id = v.visit_id
            first_htn_time = get_time(v)
        # 判断 target
        if first_target_idx is None and has_icd(v, target_pat, "diagnoses_icd"):
            first_target_idx = i
            first_target_visit_id = v.visit_id
            first_target_time = get_time(v)

    # 如果没有HTN，无法构造样本
    if first_htn_idx is None:
        return samples

    # 排除规则 #3：访视数 < 2
    if num_visits < 2:
        return samples

    # 排除规则 #1：先target后HTN
    if first_target_idx is not None and first_target_idx < first_htn_idx:
        return samples

    # 排除规则 #2：同次/同时时间
    def same_time(t1, t2):
        if (t1 is None) or (t2 is None):
            return False
        # 字符串时间规范化
        from datetime import datetime
        if isinstance(t1, str):
            t1 = datetime.fromisoformat(t1.replace('Z', '+00:00'))
        if isinstance(t2, str):
            t2 = datetime.fromisoformat(t2.replace('Z', '+00:00'))
        return t1 == t2

    simultaneous = (
        first_target_idx is not None and
        first_htn_idx is not None and (
            first_target_visit_id == first_htn_visit_id or
            same_time(first_target_time, first_htn_time)
        )
    )
    if simultaneous:
        return samples

    # -------- 构造 X（历史有效访视）与 label --------
    # 有效访视：需同时具备 诊断/手术/用药 三类代码（与你原脚本一致）
    # 历史范围：
    #   - case（HTN->target）：历史 = 首个target之前的所有有效访视
    #   - control（HTN且从未target）：历史 = 所有有效访视
    is_case = first_target_idx is not None and first_target_idx > first_htn_idx

    # 历史上限（不含上限）：case 截止首个DM；control 截止最后一次
    hist_end = first_target_idx if is_case else num_visits

    all_conditions, all_procedures, all_drugs, all_delta_days = [], [], [], []
    previous_valid_visit = None
    last_hist_valid_visit_id = None

    from datetime import datetime

    for i in range(hist_end):
        v = patient[i]
        conds = v.get_code_list(table="diagnoses_icd")
        procs = v.get_code_list(table="procedures_icd")
        meds  = v.get_code_list(table="prescriptions")

        # 必须三者都有（与你原脚本逻辑保持一致）
        # if len(conds) * len(procs) * len(meds) == 0:
        #     continue

        # delta_days
        delta_days_value = 0
        if previous_valid_visit is not None:
            try:
                t_cur = getattr(v, "encounter_time", None)
                t_prev = getattr(previous_valid_visit, "encounter_time", None)
                if t_cur and t_prev:
                    if isinstance(t_cur, str):
                        t_cur = datetime.fromisoformat(t_cur.replace('Z', '+00:00'))
                    if isinstance(t_prev, str):
                        t_prev = datetime.fromisoformat(t_prev.replace('Z', '+00:00'))
                    delta_days_value = max(0, (t_cur - t_prev).days)
                else:
                    delta_days_value = 1
            except Exception:
                delta_days_value = 1

        all_conditions.append(conds)
        all_procedures.append(procs)
        all_drugs.append(meds)
        all_delta_days.append([delta_days_value])
        last_hist_valid_visit_id = v.visit_id
        previous_valid_visit = v

    # 历史必须至少有 1 次有效访视
    if len(all_conditions) == 0:
        return samples

    # 标签与锚定 visit_id
    if is_case:
        label = 1 # HTN->target
        anchor_visit_id = first_target_visit_id  # 预测目标发生点
    else:
        label = 0  # HTN且从未发生target
        anchor_visit_id = last_hist_valid_visit_id  # 控制组用最后一个历史有效访视

    samples.append({
        "visit_id": anchor_visit_id,
        "patient_id": patient.patient_id,
        "conditions": all_conditions,
        "procedures": all_procedures,
        "drugs": all_drugs,
        "delta_days": all_delta_days,
        "label": label,
    })
    return samples



def prepare_mimic4_cormobidity_dataset():
    """
    Prepare MIMIC-IV dataset for HTN->(DM/CVD) comorbidity prediction with multiple features.
    与 prepare_mimic4_dataset 基本一致，只是换了 task_fn 和 task_name。
    """
    print("=" * 80)
    print(f"MIMIC-IV Data Preparation (HTN -> {TARGET_DISEASE} Comorbidity) using PyHealth")
    print("=" * 80)

    # Step 1: Load MIMIC-IV base dataset（保持你的参数风格）
    print("\n[Step 1/4] Loading MIMIC-IV base dataset...")
    print(f"Reading from: {MIMIC4_ROOT}")

    try:
        base_dataset = MIMIC4Dataset(
            root=MIMIC4_ROOT,
            tables=[
                "diagnoses_icd",
                "procedures_icd",
                "prescriptions",
            ],
            code_mapping=None,
            dev=False,
            refresh_cache=False
        )
        print(f"✓ Base dataset loaded successfully!")
        print(f"  - Number of patients: {len(base_dataset.patients)}")
        print(f"  - Available tables: {base_dataset.tables}")

    except Exception as e:
        print(f"✗ Error loading base dataset: {e}")
        raise

    # Step 2: Create HTN->target comorbidity task
    print(f"\n[Step 2/4] Creating HTN->{TARGET_DISEASE} comorbidity prediction task with delta_days...")
    try:
        sample_dataset = base_dataset.set_task(
            task_fn=custom_cormobidity_prediction_mimic4_fnh,
            task_name=f"htn_to_{TARGET_DISEASE.lower()}"
        )
        print(f"✓ Task created successfully!")
        print(f"  - Number of samples: {len(sample_dataset)}")

        if len(sample_dataset) > 0:
            sample = sample_dataset[0]
            print(f"\n  Sample structure:")
            print(f"    - Available keys: {sample.keys()}")
            if 'conditions' in sample:
                print(f"    - Conditions (diagnoses): {len(sample['conditions'])} visits")
            if 'procedures' in sample:
                print(f"    - Procedures: {len(sample['procedures'])} visits")
            if 'drugs' in sample:
                print(f"    - Drugs: {len(sample['drugs'])} visits")
            if 'delta_days' in sample:
                print(f"    - Delta days: {sample['delta_days']}")
            if 'label' in sample:
                print(f"    - Label: {sample['label']}")

    except Exception as e:
        print(f"✗ Error creating task: {e}")
        raise

    # Step 3: Save to pickle file（沿用你的输出目录与命名风格）
    print("\n[Step 3/4] Saving dataset to pickle file...")
    try:
        task_output_dir = os.path.join(OUTPUT_DIR, f"HTN2{TARGET_DISEASE}")
        os.makedirs(task_output_dir, exist_ok=True)
        out_path = os.path.join(
            task_output_dir, f"sample_dataset_mimiciv_htn2{TARGET_DISEASE.lower()}_multifea.pkl"
        )
        save_pickle(sample_dataset, out_path)
        print(f"✓ Dataset saved successfully!")
        print(f"  - Output path: {out_path}")
        print(f"  - File size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        raise

    # Step 4: Verify
    print("\n[Step 4/4] Verifying saved file...")
    try:
        from pyhealth.utils import load_pickle
        loaded = load_pickle(out_path)
        print(f"✓ Verification successful!")
        print(f"  - Loaded {len(loaded)} samples")
    except Exception as e:
        print(f"✗ Error verifying file: {e}")
        raise

    print("\n" + "=" * 80)
    print(f"✓ HTN->{TARGET_DISEASE} comorbidity data preparation completed successfully!")
    print("=" * 80)

    return sample_dataset


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--target",
            type=str,
            default="dm",
            choices=["dm", "cvd", "ckd", "DM", "CVD", "CKD"],
            help="Comorbidity target disease: dm/cvd/ckd (HTN -> target).",
        )
        args = parser.parse_args()
        TARGET_DISEASE = normalize_target(args.target)

        # dataset = prepare_mimic4_dataset()
        dataset = prepare_mimic4_cormobidity_dataset()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        print("\nIf you encounter issues, try:")
        print("  1. Set dev=True in MIMIC4Dataset() for testing with small subset")
        print("  2. Check PyHealth documentation: https://github.com/sunlabuiuc/PyHealth")
        print("  3. Verify MIMIC-IV data integrity")

