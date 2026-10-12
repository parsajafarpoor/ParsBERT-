"""پیدا کردن برچسب‌های احتمالاً اشتباه در train.csv با روش Confident Learning.

ایده: داده را ۵ قسمت می‌کنیم. هر بار مدل را روی ۴ قسمت آموزش می‌دهیم و دربارهٔ قسمت پنجم
که ندیده نظر می‌پرسیم. این‌طوری برای هر خبر train یک پیش‌بینی «بی‌طرف» داریم.
خبری که مدل با اطمینان بالا با برچسبش مخالف است، مشکوک است.

قانون مشکوک بودن (Northcutt et al., 2021):
    آستانهٔ هر کلاس = میانگین احتمالی که مدل به خبرهای همان کلاس داده است.
    خبری مشکوک است که احتمال یک کلاس دیگر، از آستانهٔ آن کلاس بیشتر باشد.

اجرا (روی GPU؛ ۵ بار آموزش):
    python find_label_errors.py

خروجی: suspect_labels.csv، مرتب از مشکوک‌ترین. ستون new_label را خودتان پر کنید:
    برچسب درست      ← برچسب آن خبر عوض می‌شود
    حذف             ← آن خبر از آموزش کنار گذاشته می‌شود
    خالی            ← دست نمی‌خورد
بعد: python train.py --label_fixes suspect_labels.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from train import NewsDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HooshvareLab/bert-fa-zwnj-base")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    print("دستگاه:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    df = pd.read_csv(f"{args.data_dir}/train.csv")
    labels = sorted(df["label"].unique())  # همان ترتیب train.py
    label2id = {name: i for i, name in enumerate(labels)}
    y = df["label"].map(label2id).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    enc = tokenizer(df["text"].tolist(), truncation=True, max_length=args.max_length)

    def subset(idx, with_labels=True):
        part = {k: [v[i] for i in idx] for k, v in enc.items()}
        return NewsDataset(part, [int(y[i]) for i in idx] if with_labels else None)

    # احتمال‌های «بی‌طرف»: هر خبر را مدلی پیش‌بینی می‌کند که آن را ندیده است.
    oof = np.zeros((len(df), len(labels)))
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (fit_idx, hold_idx) in enumerate(folds.split(df, y), start=1):
        print(f"---------- قسمت {k} از {args.folds} ----------")
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels))
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="outputs/label_errors_tmp",
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size * 2,
                learning_rate=args.lr,
                weight_decay=0.01,
                warmup_steps=int(0.1 * args.epochs * len(fit_idx) / args.batch_size),
                save_strategy="no",
                logging_steps=50,
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=torch.cuda.is_available(),
                report_to="none",
                seed=args.seed,
            ),
            train_dataset=subset(fit_idx),
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        trainer.train()
        logits = trainer.predict(subset(hold_idx, with_labels=False)).predictions
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        oof[hold_idx] = e / e.sum(axis=1, keepdims=True)
        print(f"دقت روی قسمت ندیده: {(oof[hold_idx].argmax(1) == y[hold_idx]).mean():.4f}")

    print(f"دقت کلی k-fold روی ۳۰۰۰ خبر: {(oof.argmax(1) == y).mean():.4f}")

    # ---------- Confident Learning ----------
    # آستانهٔ هر کلاس: میانگین احتمالی که مدل به خبرهای واقعاً همان کلاس داده است.
    thresholds = np.array([oof[y == c, c].mean() for c in range(len(labels))])
    p_given = oof[np.arange(len(df)), y]
    above = oof >= thresholds  # کدام کلاس‌ها از آستانه‌شان رد شده‌اند
    above[np.arange(len(df)), y] = False  # کلاس خود برچسب را حساب نکن
    masked = np.where(above, oof, -1)
    suspect = above.any(axis=1)
    predicted = masked.argmax(1)

    out = pd.DataFrame({
        "row": np.arange(len(df)),
        "text": df["text"],
        "label": df["label"],
        "predicted": [labels[i] for i in predicted],
        "p_label": p_given.round(3),
        "p_predicted": oof[np.arange(len(df)), predicted].round(3),
        "new_label": "",
    })[suspect].sort_values("p_label")
    out.to_csv("suspect_labels.csv", index=False, encoding="utf-8-sig")

    print(f"تعداد خبرهای مشکوک: {len(out)} از {len(df)}")
    print("بیشترین جفت‌ها (برچسب فعلی ← نظر مدل):")
    print(out.groupby(["label", "predicted"]).size().sort_values(ascending=False).head(10).to_string())
    print("suspect_labels.csv ساخته شد.")


if __name__ == "__main__":
    main()
