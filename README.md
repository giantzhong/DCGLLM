# DCGLLM: LLM-Guided Dynamic Comorbidity Graph Learning for Personalized Risk Prediction

## Directory Structure

```
DCGLLM/
├── data/                          # Preprocessed datasets
│   ├── MIMIC3/
│   │   ├── HTN2CKD/               # Hypertension → CKD task
│   │   ├── HTN2CVD/               # Hypertension → CVD task
│   │   └── HTN2DM/                # Hypertension → DM task
│   ├── MIMIC4/
│   │   ├── HTN2CKD/
│   │   ├── HTN2CVD/
│   │   └── HTN2DM/
│   ├── HuaDong/
│   │   └── HTN2DM/
│   └── README.md                  # Data acquisition instructions
├── data_preprocess/               # Data preprocessing pipeline
│   ├── prepare_mimiciii_data.py   # MIMIC-III raw data extraction
│   ├── prepare_mimiciv_data.py    # MIMIC-IV raw data extraction
│   ├── utils.py                   # Feature encoding (TF-IDF, etc.)
│   ├── data_postprocess.py        # Build graph-merged sample files
│   ├── dataset_merge.py           # Merge EHR samples with graph data
│   ├── data_config.py             # Dataset path configuration
│   ├── diagnoses.py               # Diagnosis code processing
│   ├── ehr_data.py                # EHR data structures
│   ├── node_masks.py              # Node mask generation
│   ├── tfidf_encode.py            # TF-IDF feature encoding
│   ├── code_id_map.py             # ICD code ↔ integer ID mapping
│   └── comorbidity_icdcode.py     # Comorbidity ICD definitions
├── simple_build_graph/            # LLM-based graph construction
│   ├── build_graph.py             # Perplexity-based edge scoring (multi-GPU)
│   └── build_graph_score.py       # Logit-based edge scoring
├── model/                         # Model components
│   ├── DCGLLM.py                  # Main model: fusion of graph + sequence
│   ├── GraphEncoder.py            # Static GNN encoder (GCN)
│   ├── EvolveGCN.py               # Dynamic graph encoder (DySAT variants)
│   ├── RNN.py                     # GRU-based sequence encoder
│   ├── StageNet.py                # StageNet sequence encoder
│   ├── hita_transformer.py        # HiTANet encoder
│   ├── hita_transformer_layer.py  # HiTANet attention layer
│   └── Transformer.py             # Standard Transformer encoder
├── utility/
│   └── parser_seed.py             # Argument parser (all CLI flags)
├── checkpoint/                    # Graph construction checkpoints
│   ├── MIMIC3/
│   └── MIMIC4/
├── trained_model/                 # Training result logs
│   ├── MIMIC3/
│   ├── MIMIC4/
│   └── HuaDong/
├── picture/                       # Figures and analysis notebooks
├── config.py                      # Global config (LLM model, paths, prompts)
├── loader.py                      # PyTorch DataLoader utilities
├── main_dcgllm.py                 # Training & evaluation entry point
├── data_pre.sh                    # End-to-end data preparation script
├── run_mimic3.sh                  # Run all models on MIMIC-III
├── run_mimic4.sh                  # Run all models on MIMIC-IV
├── run_huadong.sh                 # Run all models on HuaDong
├── abalation_d.sh                 # Ablation: dataset variants
├── abalation_e.sh                 # Ablation: edge weight modes
├── abalation_m.sh                 # Ablation: model variants
├── train_sizem3.sh                # Data scarcity experiments (MIMIC-III)
├── train_sizem4.sh                # Data scarcity experiments (MIMIC-IV)
├── train_sizem_huadong.sh         # Data scarcity experiments (HuaDong)
├── requirements.txt               # Python dependencies
└── data_view.ipynb                # Data exploration notebook
```

---

## Data

> The MIMIC-III and MIMIC-IV datasets require credentialed access. Apply at [physionet.org](https://mimic.physionet.org/) and see [PyHealth](https://github.com/sunlabuiuc/PyHealth) for preprocessing guidance.

Each processed task folder (`data/{DATASET}/HTN2{TARGET}/`) contains:

| File | Description |
|---|---|
| `ehr_data.json` | Patient visit sequences with ICD codes and time intervals |
| `ehr_data_trunced.json` | Truncated version for LLM prompting |
| `diagnoses.json` | Per-patient diagnosis sets |
| `code_to_id.json` | ICD code → integer node ID |
| `icd_to_id.json` | ICD string → integer ID |
| `id_to_icd.json` | Integer ID → ICD string |
| `code_to_icd_map.json` | Internal code → ICD code mapping |
| `node_masks.json` | Node masks (which codes are valid per patient) |
| `cfipf_matrix.npy` | CF-IDF node feature matrix |

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version |
|---|---|
| torch | 2.1.2 |
| torch_geometric | 2.4.0 |
| transformers | 4.31.0 |
| pyhealth | 1.1.4 |
| accelerate | 0.25.0 |

The LLM models (BioMistral-7B or HuatuoGPT-II) must be downloaded separately and placed at the path specified in `config.py`:

```python
# config.py
LLM_MODEL_NAME = "BioMistral-7B"
LLM_MODEL_PATH = f"/home/.../model/{LLM_MODEL_NAME}"
```

---

## Data Preparation

Run the full pipeline step-by-step (example for MIMIC-IV, CKD target):

```bash
# Step 1: Extract and format raw MIMIC-IV data
python data_preprocess/prepare_mimiciv_data.py --target CKD

# Step 2: Encode features (TF-IDF, node masks, etc.)
python data_preprocess/utils.py --dataset MIMIC4 --target CKD

# Step 3: Build LLM-scored comorbidity graph
python simple_build_graph/build_graph.py
# (or use logit-based scoring)
# python simple_build_graph/build_graph_score.py --dataset MIMIC4 --target CKD

# Step 4: Post-process graph data
python data_preprocess/data_postprocess.py

# Step 5: Merge EHR samples with graph embeddings
python data_preprocess/dataset_merge.py
```

Or run the reference pipeline script (see `data_pre.sh` for all variants).

Graph construction is checkpointed under `checkpoint/{DATASET}/` and can be resumed if interrupted.

---

## Training

### Quick Start

```bash
python main_dcgllm.py \
  --dataset MIMIC4 \
  --target CKD \
  --modeltype Transformer \
  --train_lr 0.0015 \
  --weight_decay 0.001 \
  --batch_size 256 \
  --epochs_train 100 \
  --use_dynamic_graph \
  --use_cross_attention \
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10
```

### Using Provided Scripts

```bash
# Run all models on MIMIC-IV (pass GPU ID as argument, default=4)
bash run_mimic4.sh 0

# Run all models on MIMIC-III
bash run_mimic3.sh 1

# Run HuaDong experiments
bash run_huadong.sh 2
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `MIMIC4` | Dataset: `MIMIC3`, `MIMIC4`, `HuaDong` |
| `--target` | `dm` | Comorbidity target: `dm`, `cvd`, `ckd` |
| `--modeltype` | `HiTANet` | Backbone: `GRU`, `StageNet`, `HiTANet`, `Transformer` |
| `--use_dynamic_graph` | off | Enable EvolveGCN dynamic graph encoder |
| `--use_cross_attention` | off | Enable cross-attention fusion module |
| `--edge_weight_mode` | `full` | Edge weights: `full`, `none`, `random`, `norm` |
| `--train_lr` | `1e-3` | Learning rate for the EHR encoder |
| `--gencoder_lr` | `1e-4` | Learning rate for the graph encoder |
| `--use_focal_loss` | off | Use Focal Loss (for class imbalance) |
| `--use_weighted_loss` | off | Use weighted BCE (for class imbalance) |
| `--use_lr_scheduler` | off | Enable LR scheduler |
| `--lr_scheduler_type` | `cosine` | Scheduler: `cosine`, `step`, `plateau`, `warmup_cosine` |
| `--eval_threshold_search` | off | Grid-search best classification threshold on val set |
| `--seed` | `[1,2,3,4,5]` | Random seeds (runs once per seed, reports mean±std) |
| `--train_val_test_split` | `[0.8,0.1,0.1]` | Dataset split ratios |

### Evaluation Metrics

The default metrics are: `accuracy`, `pr_auc`, `roc_auc`, `balanced_accuracy`, `f1`, `precision`, `mcc`.

The primary monitor metric (used for model selection) is `pr_auc`.

### Results

Training logs and results are saved to:
```
trained_model/{DATASET}/HTN2{TARGET}/{MODELTYPE}.txt
```

---

## Ablation Studies

```bash
# Edge weight ablation (full / none / random / norm)
bash abalation_e.sh

# Model backbone ablation
bash abalation_m.sh

# Dataset/task ablation
bash abalation_d.sh
```

## Data Scarcity Experiments

```bash
bash train_sizem4.sh    # MIMIC-IV
bash train_sizem3.sh    # MIMIC-III
bash train_sizem_huadong.sh  # HuaDong
```

---

## Supported LLM Backends

| Model | Graph file suffix | Notes |
|---|---|---|
| BioMistral-7B (default) | `*BioMistral-7B.pkl` | Perplexity-based or logit-based scoring |
| HuatuoGPT-II | `*HuatuoGPT_II.pkl` | Perplexity-based scoring |

Switch models by setting `LLM_MODEL_NAME` in `config.py`.

---

## Citation

If you use this code, please cite the corresponding paper (details TBD).
