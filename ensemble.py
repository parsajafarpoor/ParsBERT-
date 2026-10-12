"""ترکیب چند مدل (ensemble): میانگین احتمال‌های چند اجرای train.py.

هر اجرای train.py امتیازهایش را در پوشهٔ --output_dir خودش ذخیره می‌کند.
این اسکریپت احتمال‌های همهٔ آن پوشه‌ها را میانگین می‌گیرد.

اجرا:
    python ensemble.py outputs_s1 outputs_s2 outputs_s3

خروجی: دقت val هر مدل و دقت ترکیب (اگر val_logits موجود باشد)، و predictions_ensemble.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit("پوشه‌های خروجی مدل‌ها را بدهید، مثلاً: python ensemble.py outputs_s1 outputs_s2")

    train_df = pd.read_csv("data/train.csv")
    labels = sorted(train_df["label"].unique())  # همان ترتیب train.py

    # ---------- دقت val ِ هر مدل و ترکیبشان ----------
    if all((Path(d) / "val_logits.npy").exists() for d in dirs):
        val_df = pd.read_csv("data/val.csv")
        y_true = val_df["label"].map({name: i for i, name in enumerate(labels)}).to_numpy()
        probs = []
        for d in dirs:
            p = softmax(np.load(Path(d) / "val_logits.npy"))
            probs.append(p)
            print(f"{d}: دقت val = {(p.argmax(-1) == y_true).mean():.4f}")
        ens = np.mean(probs, axis=0).argmax(-1)
        print(f"ترکیب {len(dirs)} مدل: دقت val = {(ens == y_true).mean():.4f}")
    else:
        print("val_logits.npy در همهٔ پوشه‌ها نیست (مثلاً بعد از --train_on_all)؛ فقط test ترکیب می‌شود.")

    # ---------- ترکیب روی test ----------
    test_df = pd.read_csv("data/test.csv")
    test_probs = np.mean([softmax(np.load(Path(d) / "test_logits.npy")) for d in dirs], axis=0)
    out = pd.DataFrame({"id": test_df["id"], "label": [labels[i] for i in test_probs.argmax(-1)]})
    out.to_csv("predictions_ensemble.csv", index=False)
    print("predictions_ensemble.csv ساخته شد. توزیع برچسب‌ها:")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
