"""فاین‌تیون ParsBERT برای دسته‌بندی اخبار فارسی (۶ کلاس).

اجرا:
    python train.py                      # تنظیمات پیش‌فرض
    python train.py --epochs 4 --lr 3e-5 # تغییر تنظیمات

خروجی‌ها:
    outputs/best_model/   بهترین مدل (بر اساس دقت روی val)
    predictions.csv       پیش‌بینی برای test با ستون‌های id,label
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    # مدلی که نیم‌فاصله (ZWNJ) را درست توکن می‌کند؛ متن خبرها پر از نیم‌فاصله است.
    p.add_argument("--model", default="HooshvareLab/bert-fa-zwnj-base")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="outputs")
    # بیشتر خبرها زیر ۵۰ کلمه‌اند، پس ۱۲۸ توکن تقریباً همه را پوشش می‌دهد.
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    # نرخ یادگیری کوچک: مدل از قبل زبان فارسی را بلد است، فقط کمی تنظیمش می‌کنیم.
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    # برخی کارت‌ها (مثل سری GTX 16xx) با fp16 گاهی loss را nan می‌کنند؛ با این گزینه خاموشش کنید.
    p.add_argument("--no_fp16", action="store_true")
    # آموزش نهایی روی train+val (۲۰٪ دادهٔ بیشتر). در این حالت ارزیابی روی val معنی ندارد،
    # پس اول تنظیمات خوب را با اجرای معمولی پیدا کنید و بعد با همان تنظیمات این گزینه را بزنید.
    p.add_argument("--train_on_all", action="store_true")
    return p.parse_args()


class NewsDataset(torch.utils.data.Dataset):
    """متن‌های توکن‌شده (و در صورت وجود، برچسب‌ها) را به شکلی که Trainer می‌خواهد برمی‌گرداند."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def main():
    args = parse_args()
    # seed ثابت باعث می‌شود هر بار اجرا نتیجهٔ یکسان بدهد.
    set_seed(args.seed)
    print("دستگاه:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # ---------- ۱. خواندن داده ----------
    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/val.csv")
    test_df = pd.read_csv(f"{args.data_dir}/test.csv")

    # ---------- ۲. تبدیل برچسب متنی به عدد ----------
    # مدل فقط عدد می‌فهمد: «ورزش» -> 5 و برعکس.
    labels = sorted(train_df["label"].unique())
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}
    print("کلاس‌ها:", label2id)

    # ---------- ۳. توکن کردن ----------
    # tokenizer متن را به تکه‌ها (توکن) و سپس به شمارهٔ هر توکن در واژگان مدل تبدیل می‌کند.
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def encode(df):
        # padding اینجا انجام نمی‌شود؛ DataCollator هر batch را به اندازهٔ طولانی‌ترین متن همان batch پر می‌کند.
        return tokenizer(df["text"].tolist(), truncation=True, max_length=args.max_length)

    fit_df = pd.concat([train_df, val_df], ignore_index=True) if args.train_on_all else train_df
    train_ds = NewsDataset(encode(fit_df), fit_df["label"].map(label2id).tolist())
    val_ds = NewsDataset(encode(val_df), val_df["label"].map(label2id).tolist())
    test_ds = NewsDataset(encode(test_df))

    # ---------- ۴. ساختن مدل ----------
    # بدنهٔ ParsBERT از قبل آموزش‌دیده است؛ یک لایهٔ دسته‌بند تازه با ۶ خروجی رویش اضافه می‌شود.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    # ---------- ۵. معیار ارزیابی ----------
    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }

    # ---------- ۶. تنظیمات آموزش ----------
    # ۱۰٪ اول آموزش، نرخ یادگیری آرام بالا می‌رود تا لایهٔ تازهٔ دسته‌بند، BERT را به‌هم نریزد.
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        # بعد از هر epoch روی val ارزیابی کن (مگر وقتی val جزو دادهٔ آموزش است).
        eval_strategy="no" if args.train_on_all else "epoch",
        save_strategy="no" if args.train_on_all else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not args.train_on_all,  # در پایان، بهترین epoch را نگه دار نه آخرین را.
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available() and not args.no_fp16,  # روی GPU با دقت نیمه، سریع‌تر و کم‌حافظه‌تر.
        dataloader_pin_memory=torch.cuda.is_available(),  # pin_memory فقط روی GPU کاربرد دارد.
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if args.train_on_all else val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # ---------- ۷. آموزش ----------
    trainer.train()

    # ---------- ۸. ارزیابی نهایی روی val ----------
    if args.train_on_all:
        # امتیازهای val ِ یک اجرای قبلی را پاک کن تا balance.py نتیجهٔ کهنه گزارش نکند.
        Path(args.output_dir, "val_logits.npy").unlink(missing_ok=True)
    else:
        val_pred = trainer.predict(val_ds)
        y_true = val_pred.label_ids
        y_pred = val_pred.predictions.argmax(-1)
        np.save(f"{args.output_dir}/val_logits.npy", val_pred.predictions)
        print(val_pred.metrics)
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))

        # ماتریس درهم‌ریختگی: سطر = برچسب واقعی، ستون = پیش‌بینی مدل.
        cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=labels, columns=labels)
        print("ماتریس درهم‌ریختگی (سطر=واقعی، ستون=پیش‌بینی):")
        print(cm.to_string())

        # خبرهایی که اشتباه دسته‌بندی شدند، برای بررسی با چشم.
        wrong = val_df.assign(predicted=[id2label[i] for i in y_pred])
        wrong = wrong[wrong["label"] != wrong["predicted"]]
        wrong.to_csv("val_errors.csv", index=False, encoding="utf-8-sig")
        print("val_errors.csv ساخته شد:", len(wrong), "خبر اشتباه")

    best_dir = f"{args.output_dir}/best_model"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # ---------- ۹. پیش‌بینی test و ساخت predictions.csv ----------
    test_logits = trainer.predict(test_ds).predictions
    # امتیازهای خام را نگه می‌داریم تا balance.py بدون آموزش دوباره بتواند از آن‌ها استفاده کند.
    np.save(f"{args.output_dir}/test_logits.npy", test_logits)
    test_pred = test_logits.argmax(-1)
    out = pd.DataFrame({"id": test_df["id"], "label": [id2label[i] for i in test_pred]})
    out.to_csv("predictions.csv", index=False)
    print("predictions.csv ساخته شد:", len(out), "ردیف")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
