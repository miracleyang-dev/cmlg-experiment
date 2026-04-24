"""
Error analysis: load per-fold predictions from CMLG and baselines,
identify misclassified samples, and produce a summary.

Usage:
    python error_analysis.py \
        --data_path ./dataset/SexCommentCleaned_highconf.xlsx \
        --pred_path ./results_baselines/svm_folds_cleaned_highconf.json \
        --output_dir ./error_analysis \
        --method_name "SVM" \
        --top_n 20

For CMLG, you need to re-export per-fold predictions first
(see instructions at bottom of this file).
"""

import argparse, json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def load_predictions(pred_path):
    """Load per-fold predictions (from baselines.py output or CMLG export)."""
    with open(pred_path) as f:
        folds = json.load(f)

    all_idx, all_true, all_pred = [], [], []
    for fold_data in folds:
        all_idx.extend(fold_data["val_idx"])
        all_true.extend(fold_data["y_true"])
        all_pred.extend(fold_data["y_pred"])

    return np.array(all_idx), np.array(all_true), np.array(all_pred)


def analyze_errors(df, y_true, y_pred, indices, method_name, output_dir, top_n=20):
    """Produce error analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct = y_true == y_pred
    errors = ~correct

    print(f"\n{'='*60}")
    print(f"Error Analysis: {method_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_true)}")
    print(f"Correct: {correct.sum()} ({correct.mean()*100:.1f}%)")
    print(f"Errors:  {errors.sum()} ({errors.mean()*100:.1f}%)")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Pred=0   Pred=1")
    print(f"  True=0      {cm[0,0]:>6}   {cm[0,1]:>6}   (FP: neutral→opposing)")
    print(f"  True=1      {cm[1,0]:>6}   {cm[1,1]:>6}   (FN: opposing→neutral)")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Neutral", "Opposing"]))

    # error type breakdown
    fp_mask = (y_true == 0) & (y_pred == 1)  # false positive
    fn_mask = (y_true == 1) & (y_pred == 0)  # false negative
    print(f"False Positives (neutral misclassified as opposing): {fp_mask.sum()}")
    print(f"False Negatives (opposing misclassified as neutral): {fn_mask.sum()}")

    # extract error samples
    error_rows = []
    for i in range(len(y_true)):
        if errors[i]:
            orig_idx = indices[i]
            row = df.iloc[orig_idx]
            error_rows.append({
                "index": int(orig_idx),
                "text": row["comment_text"],
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred[i]),
                "error_type": "FP" if fp_mask[i] else "FN",
                "text_length": len(str(row["comment_text"])),
            })

    error_df = pd.DataFrame(error_rows)

    # text length distribution of errors vs correct
    correct_lengths = [len(str(df.iloc[indices[i]]["comment_text"])) for i in range(len(y_true)) if correct[i]]
    error_lengths = [r["text_length"] for r in error_rows]

    print(f"\nText length (correct): mean={np.mean(correct_lengths):.1f}, median={np.median(correct_lengths):.1f}")
    print(f"Text length (errors):  mean={np.mean(error_lengths):.1f}, median={np.median(error_lengths):.1f}")

    # save top-N error samples (shortest texts first — often hardest)
    fp_samples = error_df[error_df["error_type"] == "FP"].sort_values("text_length").head(top_n)
    fn_samples = error_df[error_df["error_type"] == "FN"].sort_values("text_length").head(top_n)

    print(f"\n--- Top {top_n} False Positive samples (neutral → opposing) ---")
    for _, r in fp_samples.iterrows():
        print(f"  [{r['index']}] {r['text'][:80]}...")

    print(f"\n--- Top {top_n} False Negative samples (opposing → neutral) ---")
    for _, r in fn_samples.iterrows():
        print(f"  [{r['index']}] {r['text'][:80]}...")

    # save full error list
    error_df.to_csv(output_dir / f"errors_{method_name}.csv", index=False, encoding="utf-8-sig")
    print(f"\nFull error list saved to {output_dir}/errors_{method_name}.csv")

    # save summary
    summary = {
        "method": method_name,
        "total": int(len(y_true)),
        "correct": int(correct.sum()),
        "errors": int(errors.sum()),
        "false_positives": int(fp_mask.sum()),
        "false_negatives": int(fn_mask.sum()),
        "avg_length_correct": float(np.mean(correct_lengths)),
        "avg_length_error": float(np.mean(error_lengths)),
    }
    with open(output_dir / f"summary_{method_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return error_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset CSV (comment_text, label)")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to per-fold predictions JSON")
    parser.add_argument("--output_dir", type=str, default="./error_analysis")
    parser.add_argument("--method_name", type=str, default="model")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    ext = Path(args.data_path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(args.data_path)
    else:
        df = pd.read_csv(args.data_path)
    # handle different label column names
    if "cleaned_label" in df.columns and "label" not in df.columns:
        df["label"] = df["cleaned_label"]
    indices, y_true, y_pred = load_predictions(args.pred_path)
    analyze_errors(df, y_true, y_pred, indices, args.method_name,
                   args.output_dir, args.top_n)


if __name__ == "__main__":
    main()


# -------------------------------------------------------------------------
# NOTE: To run error analysis on CMLG, you need to modify your ablation
# code to also save per-fold val_idx, y_true, y_pred in the same JSON
# format as baselines.py outputs:
#
#   [
#     {"fold": 0, "val_idx": [...], "y_true": [...], "y_pred": [...]},
#     {"fold": 1, ...},
#     ...
#   ]
#
# Add this to your ablation loop right after computing predictions:
#
#   fold_preds.append({
#       "fold": fold_i,
#       "val_idx": val_idx.tolist(),
#       "y_true": y_val.tolist(),
#       "y_pred": preds.tolist(),
#   })
#
# Then save:
#   with open(f"cmlg_preds_{dataset}_{setting}.json", "w") as f:
#       json.dump(fold_preds, f)
# -------------------------------------------------------------------------
