import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="DCGLLM")
    
    parser.add_argument('--seed', default=[1,2,3,4,5], nargs='*', type=int, help='Random seed.')
    parser.add_argument('--cuda_choice', nargs='?', default='cuda:0',
                        help='GPU choice.')
    parser.add_argument('--dataset', nargs='?', default='MIMIC4', choices=["MIMIC3","MIMIC4","HuaDong"],
                        help='Dataset.')
    parser.add_argument(
        "--target",
        nargs="?",
        default="dm",
        choices=["dm", "cvd", "ckd", "DM", "CVD", "CKD"],
        help="Comorbidity task target: dm/cvd/ckd (default: dm).",
    )
    parser.add_argument('--train_val_test_split', nargs='?', default='[0.8,0.1,0.1]',
                        help='Train/Val/Test Split.')
    parser.add_argument('--use_last_checkpoint', type=int, default=-1,
                        help='Use last checkpoint')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay.')
    parser.add_argument('--train_lr', type=float, default=1e-3,
                        help='Train Learning rate.')
    parser.add_argument('--monitor_criterion', default="max",
                        choices=['max','min'], nargs='?', help='Monitor_criterion.')
    

    parser.add_argument('--use_cross_attention', action='store_true',
                        help='Use cross attention for feature fusion.')
    parser.add_argument('--use_dynamic_graph', action='store_true',
                        help='Use dynamic graph encoder (EvolveGCN) for temporal graph modeling.')
    
    # Focal Loss parameters for imbalanced data
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of BCE for imbalanced data.')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal Loss alpha (weight for POSITIVE/label=1 class). For MIMIC4 (1:59 ratio), recommend 0.6-0.8.')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss gamma (focus on hard examples). Larger gamma = more focus on hard samples.')
    
    # Weighted BCE Loss (alternative to Focal Loss)
    parser.add_argument('--use_weighted_loss', action='store_true',
                        help='Use weighted BCE loss for imbalanced data.')
    parser.add_argument('--pos_weight', type=float, default=59.0,
                        help='Positive class weight for BCE. For MIMIC4 (1:59), set to 59.')
    
    # Learning rate scheduler parameters
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help='Use learning rate scheduler.')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'warmup_cosine'],
                        help='Type of learning rate scheduler.')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs for warmup_cosine scheduler.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='Learning rate decay rate for step scheduler.')
    parser.add_argument('--lr_decay_epochs', type=int, default=30,
                        help='Decay learning rate every N epochs for step scheduler.')

    parser.add_argument('--clip', type=int, default=5,
                        help='Clip Value for gradient.')
    parser.add_argument('--metrics', default=["accuracy","pr_auc","roc_auc","balanced_accuracy","f1","precision","mcc"],
                         nargs='*', help='Metrics.')
    parser.add_argument('--monitor', nargs='?', default="pr_auc",
                        help='Monitor.')
    parser.add_argument('--train_dropout_rate', type=float, default=0.1,
                        help='Train Dropout rate')

    # Evaluation-time threshold selection (for binary classification)
    parser.add_argument('--eval_threshold_metric', type=str, default='pr_auc',
                        choices=['accuracy','pr_auc', 'balanced_accuracy', 'f1', 'precision', 'recall', 'cohen_kappa', 'jaccard', 'mcc'],
                        help='Metric to optimize when selecting probability threshold on validation set.')
    parser.add_argument('--eval_threshold_search', action='store_true',
                        help='Enable grid search over probability threshold on validation set.')
    parser.add_argument('--eval_threshold_min', type=float, default=0.01,
                        help='Lower bound of threshold grid (inclusive).')
    parser.add_argument('--eval_threshold_max', type=float, default=0.99,
                        help='Upper bound of threshold grid (inclusive).')
    parser.add_argument('--eval_threshold_step', type=float, default=0.01,
                        help='Step size of threshold grid.')
    parser.add_argument('--fixed_threshold', type=float, default=0.5,
                        help='Fixed probability threshold if search is disabled.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')

    
    parser.add_argument('--modeltype', default="HiTANet",
                        choices=["GRU","StageNet","HiTANet","Transformer"], nargs='?', help='Model type.')

    
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dim.')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embed dim.')
    parser.add_argument('--encoder_layer', type=int, default=1,
                        help='Encoder layer.')

    
    parser.add_argument('--encoder_head', type=int, default=4,
                        help='Transformer Encoder Head Num.')

    
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='Chunk_Size.')
    parser.add_argument('--levels', type=int, default=2,
                        help='Levels.')

    parser.add_argument('--node_dim', type=int, default=256,
                        help='Node Embedding size.')
    parser.add_argument('--gencoder_dim_list', nargs='?', default='[256,256]',
                        help='Output sizes of every aggregation layer in Graph Encoder.')
    parser.add_argument('--gencoder_lr', type=float, default=1e-4,
                        help='Graph Encoder Learning rate.')

    # Ablation study parameter for edge weight control
    parser.add_argument('--edge_weight_mode', type=str, default='full',
                        choices=['full', 'none', 'random','norm'],
                        help='Edge weight mode for ablation study: '
                             'full=use computed edge weights, '
                             'none=no edge weights (edge_attr=None), '
                             'random=randomize edge weights.')

    parser.add_argument('--epochs_train', type=int, default=100,
                        help='Number of training epoch.')
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--max_epochs_before_stop', default=15, type=int,
                        help='stop training if dev does not increase for N epochs')

    args = parser.parse_args()

    # Normalize target -> "DM" / "CVD" / "CKD" (keep consistent with DataConfig conventions)
    t = (getattr(args, "target", "dm") or "").strip().upper()
    if t in ("DM", "CVD", "CKD"):
        target_up = t
    elif t in ("dm", "cvd", "ckd"):
        target_up = t.upper()
    else:
        raise ValueError(f"Unknown target={args.target!r}. Use 'dm', 'cvd', or 'ckd'.")
    
    # save_dir = './trained_model/{}/{}/'.format(
    #     args.dataset,
    #     str(datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    #     )
    
    save_dir = './trained_model/{}/HTN2{}/train_lr:{}/weight_decay:{}/loss:{}'.format(
        args.dataset,
        target_up,
        args.train_lr,
        args.weight_decay,
        'use_focal_loss' if args.use_focal_loss else 'use_weighted_loss'
        )
    args.save_dir = save_dir

    return args