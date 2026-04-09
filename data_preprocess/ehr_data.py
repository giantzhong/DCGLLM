import json
from tqdm import tqdm

from data_preprocess.code_id_map import lookup_icd_code, who_icd10_desc

def _lookup_disease_text(code: str, dataset: str = None):
    if dataset == "HuaDong":
        return who_icd10_desc(code)
    return lookup_icd_code(code)


def format_visit_records(conditions, delta_days, dataset: str = None):
    visit_records = []
    for i, (condition, delay) in enumerate(zip(conditions, delta_days)):
        diseases = []
        for code in condition:
            value = _lookup_disease_text(code, dataset=dataset)
            if value is None:
                continue
            
            diseases.append(value)

        # remove duplicates
        diseases = sorted(list(set(diseases)))
        
        condition_info = "\n".join(("  " + disease) for disease in diseases)
        delay_info = f"Days since last visit: {delay[0]}" if i > 0 else "First visit"
                
        format_record = f"Visit {i + 1} ({delay_info}):\n{condition_info}"
        visit_records.append(format_record)
    return "\n".join(visit_records)

def generate_ehr_data(dataset_sample, output_path, dataset: str = None):
    ehr_dict = {}
    for patient in tqdm(dataset_sample, desc="EHR data"):
        patient_id = int(patient['patient_id'])

        conditions = patient['conditions']
        delta_days = patient['delta_days']
        records = format_visit_records(conditions, delta_days, dataset=dataset)

        ehr_dict[patient_id] = records
    
    ehr_dict = dict(sorted(ehr_dict.items(), key=lambda x: x[0]))
    print("EHR data:", len(ehr_dict))

    with open(output_path, "w") as f:
        json.dump(ehr_dict, f, indent=4)

def format_lastest_visit_records(conditions, delta_days, remain_visit_num, dataset: str = None):
    visit_records = []
    total_visit_num = len(delta_days)
    for i, (condition, delay) in enumerate(zip(conditions, delta_days)):
        # filter visit
        if i + remain_visit_num < total_visit_num:
            continue
        
        diseases = []
        for code in condition:
            value = _lookup_disease_text(code, dataset=dataset)
            if value is None:
                continue
            
            diseases.append(value)

        # remove duplicates
        diseases = sorted(list(set(diseases)))
        
        condition_info = "\n".join(("  " + disease) for disease in diseases)
        delay_info = f"Days since last visit: {delay[0]}" if i > 0 else "First visit"
                
        format_record = f"Visit {i + 1} ({delay_info}):\n{condition_info}"
        visit_records.append(format_record)
    return "\n".join(visit_records)

def generate_trunced_ehr_data(dataset_sample, output_path, remain_visit_num=5, dataset: str = None):
    ehr_dict = {}
    for patient in tqdm(dataset_sample, desc="EHR trunced data"):
        patient_id = int(patient['patient_id'])

        conditions = patient['conditions']
        delta_days = patient['delta_days']
        trunced_record = format_lastest_visit_records(
            conditions, delta_days, remain_visit_num, dataset=dataset
        )

        ehr_dict[patient_id] = trunced_record
    
    ehr_dict = dict(sorted(ehr_dict.items(), key=lambda x: x[0]))
    print("EHR trunced data:", len(ehr_dict))

    with open(output_path, "w") as f:
        json.dump(ehr_dict, f, indent=4)