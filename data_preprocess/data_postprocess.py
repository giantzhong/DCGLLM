import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLM_MODEL_NAME
)
from data_preprocess.data_config import DataConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess patient graph weights (task-aware)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MIMIC3",
        choices=["MIMIC3", "MIMIC4", "HuaDong"],
        help="Dataset name (default: MIMIC3)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="dm",
        choices=["dm", "cvd", "ckd", "DM", "CVD", "CKD"],
        help="Comorbidity task target: dm/cvd/ckd (default: dm)",
    )
    parser.add_argument(
        "--input-graph-path",
        type=str,
        default=None,
        help="Override input graph path (default: data/<dataset>/HTN2<TARGET>/graph_<model>.pkl)",
    )
    parser.add_argument(
        "--output-graph-path",
        type=str,
        default=None,
        help="Override output normalized graph path (default: data/<dataset>/HTN2<TARGET>/graph_norm_<model>.pkl)",
    )
    args = parser.parse_args()

    data_config = DataConfig(args.dataset, target=args.target)
    graph_dataset_path = (
        args.input_graph_path
        if args.input_graph_path
        else os.path.join(data_config.output_path, f"graph_{LLM_MODEL_NAME}.pkl")
    )
    norm_graph_dataset_path = (
        args.output_graph_path
        if args.output_graph_path
        else os.path.join(data_config.output_path, f"graph_norm_{LLM_MODEL_NAME}.pkl")
    )

    # ensure output directory exists
    os.makedirs(os.path.dirname(norm_graph_dataset_path), exist_ok=True)

    # load dataset
    with open(graph_dataset_path, "rb") as f:
        graphs: dict = pickle.load(f)
        print(graphs)

    # extract the weight of each edge in every graph in `weights` list
    weights = []
    n_error = 0
    for edges in graphs.values():
        for edge in edges:
            (_, _, w) = edge
            print(edge)
            if w == float("inf"):
                n_error += 1
                continue
            weights.append(w)

    n_edge = len(weights)
    print(n_edge, n_error)
    print("edges num:", n_edge, "error num:", n_error, "rate:", n_edge/(n_edge + n_error))

    plt.figure(figsize=(3, 2))
    plt.hist(weights, range=(0, 100))
    plt.show()

    data = np.array(weights)
    lower_percentile = np.percentile(data, 10)
    upper_percentile = np.percentile(data, 90)

    print(f"10% lower bound: {lower_percentile}")
    print(f"10% upper bound: {upper_percentile}")

    def min_max_normal(data, lower_percentile, upper_percentile):
        if data <= lower_percentile:
            return 0
        elif data >= upper_percentile:
            return 1
        else:
            return (data - lower_percentile) / (upper_percentile - lower_percentile)

    graphs_norm = {}
    weights_norm = []
    for (patient_id, edges) in graphs.items():
        edges_norm = []
        for edge in edges:
            (u, v, w) = edge
            if w == float("inf"): # Error occured during building graph
                continue
            w_norm = 1 - min_max_normal(w, lower_percentile, upper_percentile)
            if w_norm == 0: # weak correlation
                continue
            edges_norm.append((u, v, w_norm))
            weights_norm.append(w_norm)
        if len(edges_norm) == 0:
            continue
        graphs_norm[patient_id] = edges_norm
    print(len(graphs_norm), len(graphs), len(graphs) - len(graphs_norm))

    plt.figure(figsize=(3, 2))
    plt.hist(weights_norm)
    plt.show()
    '''
        {
            "patient_id": [
                (u, v, w_norm),
                (u, v, w_norm),
                ...
            ],
            ...
        }
    '''
    with open(norm_graph_dataset_path, "wb") as f:
        pickle.dump(graphs_norm, f)


if __name__ == "__main__":
    main()