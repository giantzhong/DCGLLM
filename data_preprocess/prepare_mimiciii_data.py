"""
Script to prepare MIMIC-III dataset using PyHealth
This script generates the sample_dataset_mimiciv_mortality_multifea.pkl file
With custom task function to include delta_days (time intervals between visits)
"""

import os
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.utils import save_pickle

from comorbidity_icdcode import HTN_REGEX, get_target_regex, normalize_target

# Configuration
MIMIC3_ROOT = "/home/chenqianzhong24/dataset/MIMIC-III"
OUTPUT_DIR = "/home/chenqianzhong24/project/DCGLLM/data/MIMIC3"
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_dataset_mimiciv_mortality_multifea.pkl")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_dataset_mimiciii_htn2dm_multifea.pkl")

# Comorbidity target: "DM" or "CVD" (will be overwritten by CLI if provided)
TARGET_DISEASE = "DM"

from datetime import datetime

def same_time(t1, t2, threshold_days=1):
    """ 
    判断两个时间点是否在阈值内（默认 30 天以内），如果在阈值内返回 True，否则返回 False
    """
    if (t1 is None) or (t2 is None):
        return False

    # 字符串时间规范化
    if isinstance(t1, str):
        t1 = datetime.fromisoformat(t1.replace('Z', '+00:00'))
    if isinstance(t2, str):
        t2 = datetime.fromisoformat(t2.replace('Z', '+00:00'))

    # 计算时间差（单位：天）
    delta = abs((t1 - t2).days)
    
    # 如果时间差小于阈值，则认为是同次住院
    return delta < threshold_days

def custom_cormobidity_prediction_mimic3_fnh(patient):
    """
    Custom task function for MIMIC-III HTN->DM comorbidity prediction (ONE sample per patient)

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
    # print(patient)
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
        if first_htn_idx is None and has_icd(v, htn_pat, "DIAGNOSES_ICD"):
            first_htn_idx = i
            first_htn_visit_id = v.visit_id
            first_htn_time = get_time(v)
        # 判断 target
        if first_target_idx is None and has_icd(v, target_pat, "DIAGNOSES_ICD"):
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
    # def same_time(t1, t2):
    #     if (t1 is None) or (t2 is None):
    #         return False
    #     # 字符串时间规范化
    #     from datetime import datetime
    #     if isinstance(t1, str):
    #         t1 = datetime.fromisoformat(t1.replace('Z', '+00:00'))
    #     if isinstance(t2, str):
    #         t2 = datetime.fromisoformat(t2.replace('Z', '+00:00'))
    #     return t1 == t2

    simultaneous = (
            first_target_idx is not None and
            first_htn_idx is not None and (
                first_target_visit_id == first_htn_visit_id or
                same_time(first_target_time, first_htn_time, threshold_days=30)  # 用时间差判断
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
        conds = v.get_code_list(table="DIAGNOSES_ICD")
        procs = v.get_code_list(table="PROCEDURES_ICD")
        meds  = v.get_code_list(table="PRESCRIPTIONS")

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

def prepare_mimic3_cormobidity_dataset():
    """
    Prepare MIMIC-III dataset for HTN->(DM/CVD) comorbidity prediction with multiple features.
    与 prepare_mimic4_dataset 基本一致，只是换了 task_fn 和 task_name。
    """
    print("=" * 80)
    print(f"MIMIC-III Data Preparation (HTN -> {TARGET_DISEASE} Comorbidity) using PyHealth")
    print("=" * 80)

    # Step 1: Load MIMIC-III base dataset（保持你的参数风格）
    print("\n[Step 1/4] Loading MIMIC-III base dataset...")
    print(f"Reading from: {MIMIC3_ROOT}")

    base_dataset = MIMIC3Dataset(
        root=MIMIC3_ROOT,
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "PRESCRIPTIONS",
        ],
        code_mapping=None,
        dev=False,
        refresh_cache=False
    )
    print(f"✓ Base dataset loaded successfully!")
    print(f"  - Number of patients: {len(base_dataset.patients)}")
    print(f"  - Available tables: {base_dataset.tables}")



    # Step 2: Create HTN->target comorbidity task
    print(f"\n[Step 2/4] Creating HTN->{TARGET_DISEASE} comorbidity prediction task with delta_days...")
    sample_dataset = base_dataset.set_task(
        task_fn=custom_cormobidity_prediction_mimic3_fnh,
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

    # Step 3: Save to pickle file（沿用你的输出目录与命名风格）
    print("\n[Step 3/4] Saving dataset to pickle file...")
    try:
        task_output_dir = os.path.join(OUTPUT_DIR, f"HTN2{TARGET_DISEASE}")
        os.makedirs(task_output_dir, exist_ok=True)
        out_path = os.path.join(
            task_output_dir, f"sample_dataset_mimiciii_htn2{TARGET_DISEASE.lower()}_multifea.pkl"
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

    dataset = prepare_mimic3_cormobidity_dataset()


