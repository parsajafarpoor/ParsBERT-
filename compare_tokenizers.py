"""مقایسهٔ رفتار توکنایزر fast و slow برای سؤال‌های ۱٫۱ تا ۱٫۵.

اجرا:
    python compare_tokenizers.py
"""

import sys

import numpy as np
import pandas as pd
import tokenizers
import transformers
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")  # نمایش درست فارسی در ترمینال ویندوز

MODEL = "HooshvareLab/bert-fa-zwnj-base"
SENT = "مدل‌های زبانی بزرگ، متن‌های فارسی را بهتر از گذشته می‌فهمند."
WORD = "دانش‌آموزان"
train = pd.read_csv("data/train.csv")

print("transformers:", transformers.__version__, "| tokenizers:", tokenizers.__version__)
print("پیش‌فرض is_fast:", AutoTokenizer.from_pretrained(MODEL).is_fast)

for use_fast in [True, False]:
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=use_fast)
    print("\n" + "=" * 20, f"use_fast={use_fast} (is_fast = {tok.is_fast})", "=" * 20)

    ids = tok(SENT)["input_ids"]
    print("۱٫۱ طول جمله:", len(ids))
    print("    توکن‌ها:", tok.convert_ids_to_tokens(ids))
    print("۱٫۲ دانش‌آموزان:", tok.convert_ids_to_tokens(tok(WORD)["input_ids"]))
    print("۱٫۳ «مدل‌های»:", tok.convert_ids_to_tokens(tok("مدل‌های", add_special_tokens=False)["input_ids"]))

    lengths = np.array([len(x) for x in tok(train["text"].tolist())["input_ids"]])
    print("۱٫۴ بیشترین طول train:", lengths.max())
    print("۱٫۵ میانه:", int(np.median(lengths)), "| صدک ۹۹:", int(np.percentile(lengths, 99)),
          "| بریده‌شده با ۱۲۸:", int((lengths > 128).sum()), "| با ۶۴:", int((lengths > 64).sum()))
