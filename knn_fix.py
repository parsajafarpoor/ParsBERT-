"""اصلاح پیش‌بینی‌ها با «خبر تقریباً تکراری» در train.

یک رویداد را معمولاً چند خبرگزاری با جمله‌های تقریباً یکسان گزارش می‌کنند. اگر یک خبر test
تقریباً همان متن یک خبر train باشد، برچسب آن خبر train سرنخ بسیار مطمئنی است
(روی val: ۳۳ خبر با شباهت ۰.۴ یا بیشتر، همه هم‌برچسب نزدیک‌ترین خبر train بودند).

شباهت = اشتراک کلمه‌ها تقسیم بر اجتماع کلمه‌ها (شباهت ژاکارد)، بین ۰ و ۱.

اجرا (بعد از python train.py):
    python knn_fix.py                 # روی پیش‌بینی‌های خام مدل
    python knn_fix.py --balanced      # روی پیش‌بینی‌های balance.py

خروجی: دقت val برای چند آستانهٔ شباهت، و predictions_knn.csv با بهترین آستانه.
"""

import argparse
import re

import numpy as np
import pandas as pd

from balance import balance


def words(text):
    return set(re.findall(r"[؀-ۿ‌0-9۰-۹]+", text))


def nearest(texts, ref_sets):
    """برای هر متن، شباهت و شمارهٔ شبیه‌ترین خبر train را برمی‌گرداند."""
    sims, idx = [], []
    for t in texts:
        a = words(t)
        scores = [len(a & b) / len(a | b) if a | b else 0.0 for b in ref_sets]
        j = int(np.argmax(scores))
        sims.append(scores[j])
        idx.append(j)
    return np.array(sims), np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--balanced", action="store_true")
    args = ap.parse_args()

    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/val.csv")
    test_df = pd.read_csv(f"{args.data_dir}/test.csv")
    labels = sorted(train_df["label"].unique())  # همان ترتیب train.py
    label2id = {name: i for i, name in enumerate(labels)}
    train_sets = [words(t) for t in train_df["text"]]
    train_y = train_df["label"].map(label2id).to_numpy()

    def predict(logits):
        return balance(logits) if args.balanced else logits.argmax(-1)

    # ---------- انتخاب آستانه روی val ----------
    val_y = val_df["label"].map(label2id).to_numpy()
    val_pred = predict(np.load(f"{args.output_dir}/val_logits.npy"))
    val_sim, val_nn = nearest(val_df["text"], train_sets)
    base = (val_pred == val_y).mean()
    print(f"دقت val بدون اصلاح: {base:.4f}")
    best_th, best_acc = None, base
    for th in [0.3, 0.4, 0.5, 0.6]:
        fixed = np.where(val_sim >= th, train_y[val_nn], val_pred)
        acc = (fixed == val_y).mean()
        changed = int((fixed != val_pred).sum())
        print(f"  آستانه {th}: دقت {acc:.4f}  ({changed} پیش‌بینی عوض شد)")
        if acc > best_acc:
            best_th, best_acc = th, acc

    if best_th is None:
        print("هیچ آستانه‌ای روی val کمک نکرد؛ predictions_knn.csv ساخته نشد.")
        return

    # ---------- اعمال روی test ----------
    test_pred = predict(np.load(f"{args.output_dir}/test_logits.npy"))
    test_sim, test_nn = nearest(test_df["text"], train_sets)
    fixed = np.where(test_sim >= best_th, train_y[test_nn], test_pred)
    out = pd.DataFrame({"id": test_df["id"], "label": [labels[i] for i in fixed]})
    out.to_csv("predictions_knn.csv", index=False)
    print(f"آستانهٔ انتخابی {best_th}؛ {int((fixed != test_pred).sum())} پیش‌بینی test عوض شد.")
    print("predictions_knn.csv ساخته شد.")


if __name__ == "__main__":
    main()
