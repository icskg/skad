import argparse
import json
import warnings

from comm.early_stopping import EarlyStopping
from train.train import setup_training_environment, create_data_module, create_model_and_optimizer, model_fit, \
    validate_model

warnings.filterwarnings("ignore", category=Warning)

def parse_args():
    ap = argparse.ArgumentParser(description="KMAD Training (train.csv + test.csv + graph.json)")
    ap.add_argument('--dataset', type=str, default=None, help='Path to train.csv (no labels).')
    ap.add_argument('--win_len', type=int, default=None, help='Window length T.')
    ap.add_argument('--seg_len', type=int, default=None, help='Segment length N.')
    ap.add_argument('--stride', type=int, default=None, help='Stride between windows.')
    ap.add_argument('--batch_size', type=int, default=None, help='Batch size.')
    ap.add_argument('--patience', type=int, default=None, help='Early stopping patience.')
    ap.add_argument('--monitor', type=str, default=None, help='Metric key for early stopping.')
    ap.add_argument('--lr', type=float, default=None, help='Learning rate.')
    ap.add_argument('--val_ratio', type=float, default=None,
                    help='Fraction of train.csv reserved for validation (time-wise).')
    ap.add_argument('--max_epochs', type=int, default=None, help='Max epochs.')
    ap.add_argument('--save_best', type=str, default=None,
                    help='Path to save best-val checkpoint (.pt).')
    ap.add_argument('--eval_from', type=str, choices=[None, 'memory', 'file'], default=None,
                    help='Use in-memory weights or load from file for test metrics.')
    ap.add_argument('--load_ckpt', type=str, default=None,
                    help='Checkpoint path for evaluation when --eval_from=file. Defaults to --save_best.')

    return ap.parse_args()

def run_exp(fix_seed=True):
    args = parse_args()

    cfg, device = setup_training_environment(args, fix_seed=fix_seed)
    print(f"[INFO] start training model with configurations:{cfg}")
    dm, train_loader, val_loader, test_loader = create_data_module(cfg)
    cfg.num_vars = dm.num_vars # if drop constrained vars, num_vars will be smaller
    cfg.num_types = dm.num_types
    model, opt, scheduler = create_model_and_optimizer(cfg, device)

    early_stopping = EarlyStopping(patience=cfg.patience, mode=cfg.stop_mode, restore_best_weights=True, verbose=True, cfg=cfg)

    best_epoch = model_fit(model, train_loader, val_loader, opt, device, cfg, early_stopping, scheduler)
    metrics = validate_model(model, test_loader, device, compute_metrics=True)

    print(f"Test Metrics: Pre={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
          f"F1={metrics['f1_score']:.4f}, AUC={metrics['auc_roc']:.4f}, PR={metrics['auc_pr']:.4f}, FAR={metrics['far']:.4f} | thr={metrics['threshold']:.4f}")
    print(f"[Best] epoch={best_epoch}, saved to: {cfg.save_best}")

    with open(f'results/{cfg.dataset}_metrics.json', 'a') as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] performance metrics saved to: {f.name}")


if __name__ == "__main__":
    random_init = False
    exp_runs = 20 if random_init else 1
    for i in range(exp_runs):
        print(f"[INFO] run exp {i+1}/{exp_runs}")
        run_exp(not random_init)