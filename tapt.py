"""TAPT: ادامهٔ پیش‌آموزش ParsBERT روی متن خبرهای همین دیتاست، قبل از دسته‌بندی.

ParsBERT روی متن عمومی فارسی آموزش دیده است. اینجا همان بازی «کلمهٔ جاافتاده را حدس بزن» (MLM)
را روی متن خبرهای train و val و test انجام می‌دهیم تا مدل با سبک و واژگان خبری آشنا شود.
فقط متن خبرها استفاده می‌شود و برچسب‌ها دیده نمی‌شوند.
(Gururangan et al., 2020, "Don't Stop Pretraining")

اجرا:
    python tapt.py
    python train.py --model outputs/tapt_model
"""

import argparse

import pandas as pd
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HooshvareLab/bert-fa-zwnj-base")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="outputs/tapt_model")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    print("دستگاه:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # همهٔ متن‌ها، بدون برچسب.
    texts = pd.concat(
        [pd.read_csv(f"{args.data_dir}/{name}.csv")["text"] for name in ["train", "val", "test"]],
        ignore_index=True,
    ).tolist()
    print("تعداد متن‌ها برای TAPT:", len(texts))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = TextDataset(tokenizer(texts, truncation=True, max_length=args.max_length))
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    # ۱۵٪ توکن‌ها به‌صورت تصادفی پوشانده می‌شوند و مدل باید حدسشان بزند.
    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.output_dir + "_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=int(0.06 * args.epochs * len(dataset) / args.batch_size),
        save_strategy="no",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("مدل TAPT ذخیره شد در:", args.output_dir)
    print("حالا اجرا کنید: python train.py --model", args.output_dir)


if __name__ == "__main__":
    main()
