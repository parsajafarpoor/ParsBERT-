"""فاین‌تیون ParsBERT برای دسته‌بندی اخبار فارسی (۶ کلاس).

اجرا:
    python train.py                      # تنظیمات پیش‌فرض
    python train.py --epochs 4 --lr 3e-5 # تغییر تنظیمات

خروجی‌ها:
    outputs/best_model/   بهترین مدل (بر اساس دقت روی val)
    outputs/predictions.csv  پیش‌بینی برای test با ستون‌های id,label
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
    # فایلی که find_label_errors.py ساخته و ستون new_label آن را خودتان پر کرده‌اید.
    p.add_argument("--label_fixes", default=None)
    # همهٔ لایه‌های BERT منجمد می‌شوند و فقط classifier آموزش می‌بیند (برای مقایسه).
    # برای این حالت نرخ یادگیری بزرگ‌تری لازم است، مثلاً --lr 1e-3.
    p.add_argument("--freeze_bert", action="store_true")
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


def epoch_table(log_history):
    """برای هر دوره: میانگین loss آموزش، loss اعتبارسنجی و Macro-F1 اعتبارسنجی."""
    train_logs = [h for h in log_history if "loss" in h and "eval_loss" not in h]
    rows = []
    for h in log_history:
        if "eval_loss" not in h:
            continue
        epoch = round(h["epoch"])
        losses = [t["loss"] for t in train_logs if epoch - 1 < t["epoch"] <= epoch]
        rows.append({
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 4) if losses else float("nan"),
            "val_loss": round(h["eval_loss"], 4),
            "val_macro_f1": round(h["eval_f1_macro"], 4),
        })
    return pd.DataFrame(rows)


def plot_learning_curve(log_history, path):
    """loss آموزش (هر چند گام) و loss اعتبارسنجی (هر دوره) روی یک نمودار."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["step"], h["loss"]) for h in log_history if "loss" in h and "eval_loss" not in h]
    val = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*zip(*train), label="train loss", color="#2457c5")
    ax.plot(*zip(*val), "o-", label="validation loss", color="#c0392b")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Learning curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def apply_label_fixes(train_df, path, labels):
    """برچسب‌های اصلاح‌شده را اعمال می‌کند: برچسب جدید، «حذف»، یا خالی (بدون تغییر)."""
    fixes = pd.read_csv(path, dtype={"new_label": str}).fillna({"new_label": ""})
    fixes = fixes[fixes["new_label"].str.strip() != ""]
    drop = []
    for _, r in fixes.iterrows():
        new = r["new_label"].strip()
        if new == "حذف":
            drop.append(r["row"])
        elif new in labels:
            train_df.loc[r["row"], "label"] = new
        else:
            raise ValueError(f"برچسب ناشناخته در ردیف {r['row']}: {new!r} (باید یکی از {labels} یا «حذف» باشد)")
    print(f"اصلاح برچسب: {len(fixes) - len(drop)} خبر برچسبش عوض شد، {len(drop)} خبر حذف شد.")
    return train_df.drop(index=drop).reset_index(drop=True)


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

    if args.label_fixes:
        train_df = apply_label_fixes(train_df, args.label_fixes, labels)

    fit_df = pd.concat([train_df, val_df], ignore_index=True) if args.train_on_all else train_df
    train_ds = NewsDataset(encode(fit_df), fit_df["label"].map(label2id).tolist())
    val_ds = NewsDataset(encode(val_df), val_df["label"].map(label2id).tolist())
    test_ds = NewsDataset(encode(test_df))

    # ---------- ۴. ساختن مدل ----------
    # بدنهٔ ParsBERT از قبل آموزش‌دیده است؛ یک لایهٔ دسته‌بند تازه با ۶ خروجی رویش اضافه می‌شود.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    if args.freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"BERT منجمد شد؛ پارامترهای قابل آموزش: {n_trainable:,}")

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
        logging_steps=20,  # برای نمودار یادگیری: loss آموزش هر ۲۰ گام ثبت می‌شود
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

    # ---------- اطلاعات مدل و آموزش (برای گزارش) ----------
    n_total = sum(p.numel() for p in model.parameters())
    n_classifier = sum(p.numel() for p in model.classifier.parameters())
    n_pooler = sum(p.numel() for p in model.bert.pooler.parameters())
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    print(f"تعداد کل پارامترها: {n_total:,}")
    print(f"پارامترهای سر طبقه‌بندی (classifier): {n_classifier:,}")
    print(f"پارامترهای pooler (این checkpoint آن را ندارد و تصادفی ساخته می‌شود): {n_pooler:,}")
    print(f"گام در هر دوره: {steps_per_epoch}   کل گام‌ها: {steps_per_epoch * args.epochs}   warmup: {warmup_steps}")

    # ارزیابی پیش از هر گام آموزش: مدل هنوز چیزی یاد نگرفته است.
    if not args.train_on_all:
        before = trainer.evaluate()
        print(f"loss اعتبارسنجی پیش از آموزش: {before['eval_loss']:.4f}   (ln 6 = {math.log(len(labels)):.4f})")
        print(f"دقت پیش از آموزش: {before['eval_accuracy']:.4f}   Macro-F1: {before['eval_f1_macro']:.4f}")

    # ---------- ۷. آموزش ----------
    trainer.train()

    # جدول دوره‌ها و نمودار یادگیری از تاریخچهٔ ثبت‌شدهٔ Trainer.
    if not args.train_on_all:
        table = epoch_table(trainer.state.log_history)
        table.to_csv(f"{args.output_dir}/epochs.csv", index=False)
        print("جدول دوره‌ها:")
        print(table.to_string(index=False))
        plot_learning_curve(trainer.state.log_history, f"{args.output_dir}/learning_curve.png")
        print(f"نمودار یادگیری ذخیره شد: {args.output_dir}/learning_curve.png")

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
        wrong.to_csv(f"{args.output_dir}/val_errors.csv", index=False, encoding="utf-8-sig")
        print(f"{args.output_dir}/val_errors.csv ساخته شد:", len(wrong), "خبر اشتباه")

    best_dir = f"{args.output_dir}/best_model"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # ---------- ۹. پیش‌بینی test و ساخت predictions.csv ----------
    test_logits = trainer.predict(test_ds).predictions
    # امتیازهای خام را نگه می‌داریم تا balance.py بدون آموزش دوباره بتواند از آن‌ها استفاده کند.
    np.save(f"{args.output_dir}/test_logits.npy", test_logits)
    test_pred = test_logits.argmax(-1)
    out = pd.DataFrame({"id": test_df["id"], "label": [id2label[i] for i in test_pred]})
    # داخل پوشهٔ همین اجرا، تا اجرای بعدی پیش‌بینی‌های این مدل را بازنویسی نکند.
    out.to_csv(f"{args.output_dir}/predictions.csv", index=False)
    print(f"{args.output_dir}/predictions.csv ساخته شد:", len(out), "ردیف")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
