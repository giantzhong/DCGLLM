import os
import argparse
from pyhealth.utils import load_pickle

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocess.data_config import DataConfig
from data_preprocess.code_id_map import generate_code_id_map
from data_preprocess.diagnoses import process_patient_diagnoses_data
from data_preprocess.ehr_data import generate_ehr_data
from data_preprocess.ehr_data import generate_trunced_ehr_data
from data_preprocess.node_masks import generate_node_masks_dict
# from data_preprocess.cfipf_encode import generate_cfipf_feature
# from data_preprocess.cfipf_encode import generate_node_embeddings
from data_preprocess.tfidf_encode import generate_cfipf_feature
from data_preprocess.tfidf_encode import generate_node_embeddings

def load_dataset_from_pickle(input_path):
    mimic_sample = load_pickle(input_path)
    return mimic_sample

def preprocess_data(config: DataConfig, overwrite: bool = False):
    dataset_path = config.dataset_path
    output_path = config.output_path
    patient_dataset_path = config.patient_dataset_path
    id_to_icd_map_path = config.id_to_icd_map_path
    icd_to_id_map_path = config.icd_to_id_map_path
    code_to_icd_map_path = config.code_to_icd_map_path
    code_to_id_map_path = config.code_to_id_map_path
    ehr_data_path = config.ehr_data_path
    trunced_ehr_data_path = config.trunced_ehr_data_path
    node_masks_path = config.node_masks_path
    cfipf_dict_path = config.cfipf_dict_path
    cfipf_matrix_path = config.cfipf_matrix_path
    
    print("=" * 80)
    print(f"Dataset: {config.dataset}")
    print(f"Overwrite mode: {'ON' if overwrite else 'OFF'}")
    print("=" * 80)

    dataset_sample = None

    # create dictionary if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process patient diagnoses data
    if overwrite or not os.path.exists(patient_dataset_path):
        print(f"\n[Processing] Generating patient diagnoses data...")
        if dataset_sample is None:
            dataset_sample = load_dataset_from_pickle(dataset_path)
        process_patient_diagnoses_data(dataset_sample, patient_dataset_path, dataset=config.dataset)
        print(f"✓ Saved to: {patient_dataset_path}")
    else:
        print(f"\n[Skipping] Patient diagnoses data already exists: {patient_dataset_path}")

    # Generate code ID mappings
    if overwrite or (not os.path.exists(id_to_icd_map_path)) or (not os.path.exists(icd_to_id_map_path)) or (not os.path.exists(code_to_icd_map_path)) or (not os.path.exists(code_to_id_map_path)):
        print(f"\n[Processing] Generating code ID mappings...")
        if dataset_sample is None:
            dataset_sample = load_dataset_from_pickle(dataset_path)
        generate_code_id_map(
            dataset_sample,
            id_to_icd_map_path,
            icd_to_id_map_path,
            code_to_icd_map_path,
            code_to_id_map_path,
            dataset=config.dataset,
        )
        print(f"✓ Saved mappings to: {output_path}")
    else:
        print(f"\n[Skipping] Code ID mappings already exist")

    # Generate EHR data
    if overwrite or not os.path.exists(ehr_data_path):
        print(f"\n[Processing] Generating EHR data...")
        if dataset_sample is None:
            dataset_sample = load_dataset_from_pickle(dataset_path)
        generate_ehr_data(dataset_sample, ehr_data_path, dataset=config.dataset)
        generate_trunced_ehr_data(
            dataset_sample, trunced_ehr_data_path, remain_visit_num=5, dataset=config.dataset
        )
        print(f"✓ Saved to: {ehr_data_path}")
        print(f"✓ Saved truncated to: {trunced_ehr_data_path}")
    else:
        print(f"\n[Skipping] EHR data already exists: {ehr_data_path}")

    # Generate node masks
    if overwrite or not os.path.exists(node_masks_path):
        print(f"\n[Processing] Generating node masks...")
        generate_node_masks_dict(patient_dataset_path, icd_to_id_map_path, node_masks_path)
        print(f"✓ Saved to: {node_masks_path}")
    else:
        print(f"\n[Skipping] Node masks already exist: {node_masks_path}")

    # Generate CFIPF features
    if overwrite or not os.path.exists(cfipf_dict_path):
        print(f"\n[Processing] Generating CFIPF features...")
        if dataset_sample is None:
            dataset_sample = load_dataset_from_pickle(dataset_path)
        generate_cfipf_feature(dataset_sample, node_masks_path, id_to_icd_map_path, code_to_icd_map_path, cfipf_dict_path)
        print(f"✓ Saved to: {cfipf_dict_path}")
    else:
        print(f"\n[Skipping] CFIPF features already exist: {cfipf_dict_path}")

    # Generate node embeddings
    if overwrite or not os.path.exists(cfipf_matrix_path):
        print(f"\n[Processing] Generating node embeddings...")
        generate_node_embeddings(id_to_icd_map_path, cfipf_matrix_path)
        print(f"✓ Saved to: {cfipf_matrix_path}")
    else:
        print(f"\n[Skipping] Node embeddings already exist: {cfipf_matrix_path}")
    
    print("\n" + "=" * 80)
    print("✓ Data preprocessing completed!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for DCGLLM")
    parser.add_argument('--dataset', type=str, default='MIMIC4', 
                        choices=['MIMIC3', 'MIMIC4', 'HuaDong'],
                        help='Dataset to preprocess (default: MIMIC4)')
    parser.add_argument(
        '--target',
        type=str,
        default='dm',
        choices=['dm', 'cvd', 'ckd', 'DM', 'CVD', 'CKD'],
        help='Comorbidity task target: dm/cvd/ckd (default: dm)',
    )
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files instead of skipping them')
    parser.add_argument('--mimic4-data-path', type=str, default=None,
                        help='Override default MIMIC4 pickle path.')
    parser.add_argument('--mimic3-data-path', type=str, default=None,
                        help='Override default MIMIC3 pickle path.')
    parser.add_argument('--huadong-data-path', type=str, default=None,
                        help='Override default HuaDong pickle path.')
    
    args = parser.parse_args()
    
    # Load configuration for the specified dataset
    config = DataConfig(args.dataset, target=args.target)
    if args.mimic4_data_path and config.dataset == "MIMIC4":
        config.dataset_path = args.mimic4_data_path
        config.mimic4_dataset_path = args.mimic4_data_path
    elif args.mimic3_data_path and config.dataset == "MIMIC3":
        config.dataset_path = args.mimic3_data_path
        config.mimic3_dataset_path = args.mimic3_data_path
    elif args.huadong_data_path and config.dataset == "HuaDong":
        config.dataset_path = args.huadong_data_path
        config.huadong_dataset_path = args.huadong_data_path
    # Run preprocessing
    preprocess_data(config, overwrite=args.overwrite)
