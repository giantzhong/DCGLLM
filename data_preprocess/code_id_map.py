import json
from tqdm import tqdm
from pyhealth.medcode import InnerMap
import simple_icd_10 as icd10


icd9cm: InnerMap = InnerMap.load("ICD9CM")
icd10cm: InnerMap = InnerMap.load("ICD10CM")

def who_icd10_desc(code: str):
    if code is None:
        return None
    code = str(code).strip().upper()
    code = code.replace(".", "")  # 这个库通常用无点格式（不确定时就两种都试）
    if icd10.is_valid_item(code):
        return icd10.get_description(code)
    return None


def lookup_icd_code(code: str):# 输入ICD诊断编码查询对应的文本描述
    # look up ICD CM code
    try:
        value = icd9cm.lookup(code)
    except Exception:
        try:
            value = icd10cm.lookup(code)
        except Exception:
            value = None
    return value

def generate_code_id_map(
    dataset_sample,
    id_to_icd_map_path,
    icd_to_id_map_path,
    code_to_icd_map_path,
    code_to_id_map_path,
    dataset: str = None,
):
    icd_set = set()
    code_to_icd_map = {}
    for patient in tqdm(dataset_sample, desc="code map"):
        conditions = patient['conditions']
        for condition in conditions:
            for code in condition:
                # HuaDong uses WHO ICD-10 description mapping; keep existing behavior for others
                if dataset == "HuaDong":
                    value = who_icd10_desc(code)
                else:
                    value = lookup_icd_code(code)
                if value is None:
                    continue

                icd_set.add(value)
                code_to_icd_map[code] = value

    # sort icd set
    icd_list = list(icd_set)
    icd_list.sort()

    # create icd map
    icd_map = {}
    icd_map_reverse = {}
    for i in range(len(icd_list)):
        icd_map[icd_list[i]] = i
        icd_map_reverse[i] = icd_list[i]

    # save data to json
    with open(id_to_icd_map_path, "w") as f:
        json.dump(icd_map_reverse, f, indent=4)

    with open(icd_to_id_map_path, "w") as f:
        json.dump(icd_map, f, indent=4)

    # sort ICD CODE map
    code_to_icd_map = dict(sorted(code_to_icd_map.items(), key=lambda x: x[0]))

    with open(code_to_icd_map_path, "w") as f:
        json.dump(code_to_icd_map, f, indent=4)

    # generate code_to_id_map: code -> id (via code -> icd -> id)
    code_to_id_map = {}
    for code, icd_text in code_to_icd_map.items():
        if icd_text in icd_map:
            code_to_id_map[code] = icd_map[icd_text]
    
    # sort code_to_id_map
    code_to_id_map = dict(sorted(code_to_id_map.items(), key=lambda x: x[0]))
    
    with open(code_to_id_map_path, "w") as f:
        json.dump(code_to_id_map, f, indent=4)

    print("icd dict length: ", len(icd_map))
    print("code_to_id dict length: ", len(code_to_id_map))
