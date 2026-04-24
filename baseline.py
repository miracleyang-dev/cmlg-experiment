"""
Baseline experiments: SVM+TF-IDF and BERT-base-chinese fine-tuning.
Run on the same 3 dataset versions with the same 5-fold stratified CV.

Usage:
    python baselines.py --data_dir ./dataset --output_dir ./results_baselines

Expected data files in data_dir:
    - SexCommentNew.xlsx              (original, columns: comment_text, label)
    - SexCommentCleaned_full.xlsx     (cleaned full)
    - SexCommentCleaned_highconf.xlsx (cleaned high-confidence)
"""

import argparse, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd  # requires openpyxl: pip install openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. SVM + TF-IDF
# ---------------------------------------------------------------------------

def run_svm_tfidf(texts, labels, n_splits=5, seed=42):
    """Stratified 5-fold CV with LinearSVC + TF-IDF (char-level bigrams)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        t0 = time.time()
        X_train_text = [texts[i] for i in train_idx]
        X_val_text   = [texts[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val   = labels[val_idx]

        # char-level bigram TF-IDF — works well for Chinese without tokenizer
        tfidf = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 3), max_features=50000,
            sublinear_tf=True,
        )
        X_train = tfidf.fit_transform(X_train_text)
        X_val   = tfidf.transform(X_val_text)

        clf = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=3)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        fold_results.append({
            "fold": fold,
            "acc": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="macro"),
            "recall": recall_score(y_val, y_pred, average="macro"),
            "macro_f1": f1_score(y_val, y_pred, average="macro"),
            "f1_c0": f1_score(y_val, y_pred, pos_label=0),
            "f1_c1": f1_score(y_val, y_pred, pos_label=1),
            "time_sec": time.time() - t0,
            # keep per-fold predictions for error analysis
            "val_idx": val_idx.tolist(),
            "y_true": y_val.tolist(),
            "y_pred": y_pred.tolist(),
        })
        print(f"  SVM fold {fold}: macro_f1={fold_results[-1]['macro_f1']:.4f}  ({fold_results[-1]['time_sec']:.1f}s)")

    return fold_results


# ---------------------------------------------------------------------------
# 2. BERT-base-chinese fine-tuning
# ---------------------------------------------------------------------------

def run_bert_finetune(texts, labels, n_splits=5, seed=42,
                      model_name="bert-base-chinese", local_model_path="./embedding_models/baseline_bert",
                      epochs=5, batch_size=32, lr=2e-5, max_len=128):
    """Stratified 5-fold CV with HuggingFace BERT fine-tuning."""
    import os
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import BertTokenizer, BertForSequenceClassification, AdamW

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  BERT device: {device}")

    # Fallback to model hub if local path does not exist
    load_path = local_model_path if os.path.exists(local_model_path) else model_name
    print(f"  Loading BERT from: {load_path}")

    tokenizer = BertTokenizer.from_pretrained(load_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        t0 = time.time()
        train_texts = [texts[i] for i in train_idx]
        val_texts   = [texts[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val   = labels[val_idx]

        # tokenize
        train_enc = tokenizer(train_texts, max_length=max_len, padding="max_length",
                              truncation=True, return_tensors="pt")
        val_enc   = tokenizer(val_texts, max_length=max_len, padding="max_length",
                              truncation=True, return_tensors="pt")

        train_ds = TensorDataset(
            train_enc["input_ids"], train_enc["attention_mask"],
            torch.tensor(y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            val_enc["input_ids"], val_enc["attention_mask"],
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size)

        # model
        model = BertForSequenceClassification.from_pretrained(load_path, num_labels=2)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_f1 = 0.0
        best_preds = None

        for epoch in range(epochs):
            # train
            model.train()
            for batch in train_loader:
                input_ids, attn_mask, lbl = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=lbl)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # eval
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attn_mask, lbl = [b.to(device) for b in batch]
                    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(lbl.cpu().numpy())

            ep_f1 = f1_score(all_labels, all_preds, average="macro")
            if ep_f1 > best_f1:
                best_f1 = ep_f1
                best_preds = list(all_preds)

        y_pred = np.array(best_preds)
        fold_results.append({
            "fold": fold,
            "acc": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="macro"),
            "recall": recall_score(y_val, y_pred, average="macro"),
            "macro_f1": f1_score(y_val, y_pred, average="macro"),
            "f1_c0": f1_score(y_val, y_pred, pos_label=0),
            "f1_c1": f1_score(y_val, y_pred, pos_label=1),
            "time_sec": time.time() - t0,
            "val_idx": val_idx.tolist(),
            "y_true": y_val.tolist(),
            "y_pred": y_pred.tolist(),
        })
        print(f"  BERT fold {fold}: macro_f1={fold_results[-1]['macro_f1']:.4f}  ({fold_results[-1]['time_sec']:.1f}s)")

        # free GPU memory
        del model, optimizer
        torch.cuda.empty_cache()

    return fold_results


# ---------------------------------------------------------------------------
# 3. Aggregate & save
# ---------------------------------------------------------------------------

def aggregate(fold_results, method_name):
    """Compute mean/std from per-fold results."""
    metrics = ["acc", "macro_f1", "f1_c0", "f1_c1", "precision", "recall"]
    summary = {"method": method_name, "folds": len(fold_results)}
    # keep per-fold scores for significance tests later
    summary["per_fold_macro_f1"] = [r["macro_f1"] for r in fold_results]
    summary["per_fold_acc"]      = [r["acc"] for r in fold_results]
    for m in metrics:
        vals = [r[m] for r in fold_results]
        summary[f"{m}_mean"] = float(np.mean(vals))
        summary[f"{m}_std"]  = float(np.std(vals))
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="./dataset")
    parser.add_argument("--output_dir", type=str, default="./results_baselines")
    parser.add_argument("--skip_bert",  action="store_true", help="skip BERT if no GPU")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {
        "original":         "SexCommentNew.xlsx",
        "cleaned_full":     "SexCommentCleaned_full.xlsx",
        "cleaned_highconf": "SexCommentCleaned_highconf.xlsx",
    }

    all_results = {}

    for ds_name, filename in datasets.items():
        filepath = Path(args.data_dir) / filename
        if not filepath.exists():
            print(f"[SKIP] {filepath} not found")
            continue

        df = pd.read_excel(filepath)
        # handle different label column names across dataset versions
        if "cleaned_label" in df.columns and "label" not in df.columns:
            df["label"] = df["cleaned_label"]
        texts  = df["comment_text"].astype(str).tolist()
        labels = df["label"].values.astype(int)
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({len(texts)} samples)")
        print(f"{'='*60}")

        ds_results = {}

        # --- SVM ---
        print("\n[SVM + TF-IDF]")
        svm_folds = run_svm_tfidf(texts, labels)
        svm_summary = aggregate(svm_folds, "SVM+TF-IDF")
        ds_results["svm"] = svm_summary
        print(f"  => macro_f1 = {svm_summary['macro_f1_mean']:.4f} ± {svm_summary['macro_f1_std']:.4f}")

        # save per-fold predictions for error analysis
        with open(out / f"svm_folds_{ds_name}.json", "w") as f:
            json.dump(svm_folds, f, ensure_ascii=False, indent=2)

        # --- BERT ---
        if not args.skip_bert:
            print("\n[BERT-base-chinese fine-tune]")
            bert_folds = run_bert_finetune(texts, labels)
            bert_summary = aggregate(bert_folds, "BERT-base-chinese")
            ds_results["bert"] = bert_summary
            print(f"  => macro_f1 = {bert_summary['macro_f1_mean']:.4f} ± {bert_summary['macro_f1_std']:.4f}")

            with open(out / f"bert_folds_{ds_name}.json", "w") as f:
                json.dump(bert_folds, f, ensure_ascii=False, indent=2)

        all_results[ds_name] = ds_results

    # save aggregated summary
    with open(out / "baseline_summary.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()
