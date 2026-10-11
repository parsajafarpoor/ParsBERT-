"""تصحیح سوگیری مدل به سمت یک کلاس، با فرض اینکه داده‌ها متعادل‌اند.

مدل گاهی یک کلاس را بیش از حد پیش‌بینی می‌کند (مثلاً «بین‌الملل» به‌جای «سیاست»).
اگر بدانیم هر کلاس تقریباً سهم برابری دارد، می‌توانیم امتیاز کلاس‌ها را طوری جابه‌جا کنیم
که تعداد پیش‌بینی‌ها برابر شود. فقط خبرهای مرزی (که مدل بینشان مردد بوده) عوض می‌شوند.

اجرا (بعد از python train.py):
    python balance.py

خروجی:
    دقت val قبل و بعد از تصحیح (برای اینکه ببینیم واقعاً کمک می‌کند یا نه)
    predictions_balanced.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def balance(logits, n_iter=100):
    """احتمال‌ها را آن‌قدر تنظیم می‌کند که جمع احتمال هر کلاس برابر شود (الگوریتم Sinkhorn)."""
    p = softmax(logits)
    target = len(p) / p.shape[1]
    for _ in range(n_iter):
        p = p * (target / p.sum(axis=0, keepdims=True))  # هر ستون (کلاس) به سهم برابر
        p = p / p.sum(axis=1, keepdims=True)  # هر سطر (خبر) دوباره جمعش ۱ شود
    return p.argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    args = ap.parse_args()

    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    labels = sorted(train_df["label"].unique())  # همان ترتیب train.py

    # ---------- آزمایش روی val: آیا تصحیح دقت را بالا می‌برد؟ ----------
    val_df = pd.read_csv(f"{args.data_dir}/val.csv")
    val_logits = np.load(f"{args.output_dir}/val_logits.npy")
    y_true = val_df["label"].map({name: i for i, name in enumerate(labels)}).to_numpy()
    before = accuracy_score(y_true, val_logits.argmax(-1))
    after = accuracy_score(y_true, balance(val_logits))
    print(f"دقت val قبل از تصحیح: {before:.4f}")
    print(f"دقت val بعد از تصحیح: {after:.4f}")

    # ---------- اعمال روی test ----------
    test_df = pd.read_csv(f"{args.data_dir}/test.csv")
    test_logits = np.load(f"{args.output_dir}/test_logits.npy")
    old = test_logits.argmax(-1)
    new = balance(test_logits)
    out = pd.DataFrame({"id": test_df["id"], "label": [labels[i] for i in new]})
    out.to_csv("predictions_balanced.csv", index=False)
    print("تعداد خبرهای test که برچسبشان عوض شد:", int((old != new).sum()))
    print("predictions_balanced.csv ساخته شد. توزیع برچسب‌ها:")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
