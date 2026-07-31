from pathlib import Path
from typing import List, Tuple, Union

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

    tp_probs = [float(prob[i, j]) for i, j in tp_pairs]
    fp_probs = [float(prob[i, j]) for i, j in fp_pairs]
    fn_probs = [float(prob[i, j]) for i, j in fn_pairs]

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
    ax.hist(values, bins=30)
    mean = values.mean()
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    for v, label in [(mean, "mean"), (q25, "Q25"), (q50, "Q50"), (q75, "Q75")]:
        ax.axvline(v, linestyle="--", label=f"{label}={v:.3f}")
    ax.set_title(title)
    ax.legend(fontsize=8)


def _plot_diagnostics(fscores, precisions, recalls, tp_probs, fp_probs, fn_probs):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _hist_with_ticks(axes[0, 0], fscores, "F-score")
    _hist_with_ticks(axes[0, 1], precisions, "Precision")
    _hist_with_ticks(axes[0, 2], recalls, "Recall")

    for ax, probs, title in zip(
        axes[1], (tp_probs, fp_probs, fn_probs), ("TP probs", "FP probs", "FN probs")
    ):
        if len(probs) > 0:
            ax.hist(probs, bins=30)
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
        _plot_threshold_sweep(thresholds, results, best_threshold)
        fscores, precisions, recalls, tp_probs, fp_probs, fn_probs = results[best_threshold]

    print(
        f"threshold={best_threshold}  fscore={fscores.mean():.4f}  "
        f"precision={precisions.mean():.4f}  recall={recalls.mean():.4f}"
    )
    _plot_diagnostics(fscores, precisions, recalls, tp_probs, fp_probs, fn_probs)

    return fscores, precisions, recalls
