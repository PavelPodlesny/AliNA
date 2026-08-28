from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import numpy as np
import torch
import naskit as nsk
import matplotlib.pyplot as plt

import alina
from alina import AlinaDataset


def _quantize_to_na(model, prob, threshold, na):
    adj = model.quantize_matrix(prob, threshold).numpy()
    return nsk.NucleicAcid.from_adjacency(adj, seq=na.seq, name=na.name, meta=na.meta)


def _pair_metrics(target_na, pred_na, prob):
    target_pairs = set(target_na.pairs)
    pred_pairs = set(pred_na.pairs)

    tp_pairs = target_pairs & pred_pairs
    fp_pairs = pred_pairs - target_pairs
    fn_pairs = target_pairs - pred_pairs

    tp, fp, fn = len(tp_pairs), len(fp_pairs), len(fn_pairs)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    fscore = 2 * tp / (2 * tp + fn + fp + 1e-7)

    tp_probs = [sym_max(prob, i, j) for i, j in tp_pairs]
    fp_probs = [sym_max(prob, i, j) for i, j in fp_pairs]
    fn_probs = [sym_max(prob, i, j) for i, j in fn_pairs]

    return fscore, precision, recall, tp_probs, fp_probs, fn_probs


def _eval_at_threshold(model, target_nas, probs, threshold):
    fscores, precisions, recalls = [], [], []
    tp_probs, fp_probs, fn_probs = [], [], []

    for target_na, prob in zip(target_nas, probs):
        pred_na = _quantize_to_na(model, prob, threshold, target_na)
        fscore, precision, recall, tpp, fpp, fnp = _pair_metrics(target_na, pred_na, prob)
        fscores.append(fscore)
        precisions.append(precision)
        recalls.append(recall)
        tp_probs += tpp
        fp_probs += fpp
        fn_probs += fnp

    return (np.array(fscores), np.array(precisions), np.array(recalls),
            tp_probs, fp_probs, fn_probs)


def _hist_with_ticks(ax, values, title):
    values = np.asarray(values)
    ax.hist(values, bins=50)
    mean = values.mean()
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    for v, label, c in [(mean, "mean", "red"), (q25, "Q25", "green"), (q50, "Q50", "green"), (q75, "Q75", "green")]:
        ax.axvline(v, linestyle="--", label=f"{label}={v:.3f}", color=c)

    xticks = np.round(np.arange(0,1.01,0.1),1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, ha='right')
    
    ax.set_title(title)
    ax.set_ylabel('Counts')
    ax.legend(fontsize=8)


def _plot_diagnostics(fscores, precisions, recalls, tp_probs, fp_probs, fn_probs, best_threshold):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row', sharex='col')
    _hist_with_ticks(axes[0, 0], fscores, "F-score")
    _hist_with_ticks(axes[0, 1], precisions, "Precision")
    _hist_with_ticks(axes[0, 2], recalls, "Recall")

    for ax, probs, title in zip(
        axes[1], (tp_probs, fp_probs, fn_probs), ("TP probs", "FP probs", "FN probs")
    ):
        if len(probs) > 0:
            ax.hist(probs, bins=50, range=(0.0,1.0))
            ax.axvline(best_threshold, linestyle="--", label=f"quant th={best_threshold}", color='red')
            xticks = np.round(np.arange(0,1.01,0.1),1)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, rotation=45, ha='right')
            ax.set_ylabel('Counts')
            
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


def _plot_threshold_sweep(thresholds, results, best_threshold):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, [results[t][0].mean() for t in thresholds], marker="o", label="fscore")
    ax.plot(thresholds, [results[t][1].mean() for t in thresholds], marker="o", label="precision")
    ax.plot(thresholds, [results[t][2].mean() for t in thresholds], marker="o", label="recall")
    ax.axvline(best_threshold, color="red", linestyle="--", label=f"best={best_threshold}")
    ax.set_xlabel("threshold")
    ax.legend()
    fig.tight_layout()
    plt.show()


def evaluate(
    checkpoint_path: Union[str, Path],
    dataset_path: Union[str, Path],
    threshold: Union[float, List[float]] = 0.5,
    batch_size: int = 8,
    device: Union[str, torch.device] = "cpu",
    plot: bool = True,
    with_probs: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    model = alina.AliNA.load(path=checkpoint_path)
    model = model.to(device)

    ds = AlinaDataset.load(path=dataset_path)
    target_nas = ds.nas

    thresholds = threshold if isinstance(threshold, list) else [threshold]

    _, probs = model.fold(
        target_nas, threshold=thresholds[0], with_probs=True,
        batch_size=batch_size, verbose=True
    )

    if len(thresholds) == 1:
        best_threshold = thresholds[0]
        fscores, precisions, recalls, tp_probs, fp_probs, fn_probs = _eval_at_threshold(
            model, target_nas, probs, best_threshold
        )
    else:
        results = {t: _eval_at_threshold(model, target_nas, probs, t) for t in thresholds}
        best_threshold = max(results, key=lambda t: results[t][0].mean())
        if plot:
            _plot_threshold_sweep(thresholds, results, best_threshold)
        fscores, precisions, recalls, tp_probs, fp_probs, fn_probs = results[best_threshold]

    print(
        f"threshold={best_threshold}  fscore={fscores.mean():.4f}  "
        f"precision={precisions.mean():.4f}  recall={recalls.mean():.4f}"
    )
    if plot:
        _plot_diagnostics(fscores, precisions, recalls, tp_probs, fp_probs, fn_probs, best_threshold)

    if with_probs:
        output = (fscores, precisions, recalls, tp_probs, fp_probs, fn_probs)
    else:
        output = (fscores, precisions, recalls)
        
    return output

def sym_max(m, i, j):
    return max(float(m[i, j]), float(m[j, i]))

def evaluate_models(
    checkpoints: dict[str, Path],
    datasets: dict[str, Path],
    dataset2models: dict[str, list[str]],
    device: str,
    results_path: Path,
    th: float = 0.5,
) -> pd.DataFrame:
    """
    Run each model checkpoint on every dataset it's allowed to see, and compile a metrics dataframe.
 
    Args:
        checkpoints: {model_label: path_to_checkpoint}
        datasets: {dataset_label: path_to_dataset}
        dataset2models: {dataset_label: [model_labels allowed on this dataset]}
                         A dataset is skipped for any model_label not listed here.
        device: device string passed to evaluate(), e.g. 'cuda:0'
        results_path: where to save the resulting dataframe (as .csv)
        th: quantization threshold passed only to evaluate(); NOT used for the
            Fscore>=0.8 / Fscore<0.5 columns, which are fixed regardless of th
 
    Returns:
        DataFrame indexed by model_label ('train protocol'), with one row per (dataset, model) pair
        that was actually evaluated.
    """
    metric_cols = [
        "Fscore==1.0",
        "Fscore==0.0",
        "Fscore>=0.8",
        "Fscore<0.5",
    ]
    df_template_keys = [
        "dataset",
        "train protocol",
        "quant TH",
        "n",
        "mean precision",
        "mean recall",
        "mean Fscore",
        "median Fscore",
        *metric_cols,
    ]
 
    raw_results: dict[str, dict[str, list]] = {}
 
    for ds_label, allowed_models in dataset2models.items():
        ds_path = datasets.get(ds_label)
        if ds_path is None:
            print(f"Warning: dataset '{ds_label}' not found in `datasets`, skipping.")
            continue
 
        print(f"\n{'#' * 100}\ndataset: {ds_label}\n{'#' * 100}\n")
        raw_results[ds_label] = {key: [] for key in df_template_keys}
 
        for model_label in allowed_models:
            ckpt_path = checkpoints.get(model_label)
            if ckpt_path is None:
                print(f"Warning: model '{model_label}' not found in `checkpoints`, skipping "
                      f"for dataset '{ds_label}'.")
                continue
 
            print(f"\nmodel: {model_label}\npath: {ds_path}\n")
 
            fs, pr, rc = evaluate(
                ckpt_path, ds_path,
                threshold=th, device=device,
                plot=False, with_probs=False,
            )
 
            print(f"{'-' * 100}\n")
 
            r = raw_results[ds_label]
            r["dataset"].append(ds_label)
            r["train protocol"].append(model_label)
            r["quant TH"].append(th)
            r["n"].append(fs.shape[0])
            r["mean precision"].append(float(np.round(np.mean(pr), 3)))
            r["mean recall"].append(float(np.round(np.mean(rc), 3)))
            r["mean Fscore"].append(float(np.round(np.mean(fs), 3)))
            r["median Fscore"].append(float(np.round(np.median(fs), 3)))
            r["Fscore==1.0"].append(int(np.sum(fs == 1.0)))
            r["Fscore==0.0"].append(int(np.sum(fs == 0.0)))
            r["Fscore>=0.8"].append(int(np.sum(fs >= 0.8)))
            r["Fscore<0.5"].append(int(np.sum(fs < 0.5)))
 
    dfs = []
    for ds_label, metrics_dict in raw_results.items():
        if not metrics_dict["dataset"]:
            continue  # no model was actually evaluated on this dataset
        df = pd.DataFrame(metrics_dict)
        df = df.set_index("train protocol")
        dfs.append(df)
 
    if not dfs:
        raise ValueError("No (dataset, model) pairs were evaluated — check `dataset2models`, "
                          "`checkpoints`, and `datasets` for mismatched labels.")
 
    results = pd.concat(dfs, axis=0)
 
    for col in metric_cols:
        results[f"{col}, pct"] = (results[col] / results["n"] * 100).round(2)
 
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path)
 
    return results
