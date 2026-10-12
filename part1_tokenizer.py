"""بخش ۱ تکلیف: بررسی توکنایزر HooshvareLab/bert-fa-zwnj-base.

اجرا:
    python part1_tokenizer.py
"""

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")


def show(text):
    # تنظیمات پیش‌فرض: [CLS] و [SEP] اضافه می‌شوند.
    ids = tokenizer(text)["input_ids"]
    print("متن:", text)
    print("input_ids:", ids)
    print("توکن‌ها:", tokenizer.convert_ids_to_tokens(ids))
    print("طول input_ids:", len(ids))
    print()


print("========== سؤال ۱٫۱ و ۱٫۳ ==========")
show("مدل‌های زبانی بزرگ، متن‌های فارسی را بهتر از گذشته می‌فهمند.")

print("========== سؤال ۱٫۲ ==========")
show("دانش‌آموزان")

print("========== سؤال ۱٫۴ و ۱٫۵ ==========")
train = pd.read_csv("data/train.csv")
lengths = np.array([len(ids) for ids in tokenizer(train["text"].tolist())["input_ids"]])
print("بیشترین طول توکنی (با توکن‌های ویژه):", lengths.max())
print("کمترین:", lengths.min(), "  میانگین:", round(lengths.mean(), 1), "  میانه:", int(np.median(lengths)))
for q in [90, 95, 99]:
    print(f"صدک {q}:", int(np.percentile(lengths, q)))
for limit in [32, 64, 96, 128]:
    cut = (lengths > limit).sum()
    print(f"اگر max_length={limit}: {cut} متن ({100 * cut / len(lengths):.1f}٪) بریده می‌شوند")
print("بلندترین متن:", train["text"][lengths.argmax()])
