"""اجرای توکنایزر ParsBERT روی چند جمله و نمایش خروجی.

اجرا:
    python tokenizer_demo.py

جمله‌های تکلیف را در لیست SENTENCES بگذارید.
"""

from transformers import AutoTokenizer

MODEL = "HooshvareLab/bert-fa-zwnj-base"

# جمله‌های خواسته‌شده در تکلیف را اینجا بنویسید.
SENTENCES = [
    "تیم ملی فوتبال ایران امروز بازی کرد.",
    "بانک مرکزی نرخ بهره را افزایش داد.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)

for sentence in SENTENCES:
    # تنظیمات پیش‌فرض: add_special_tokens=True، پس [CLS] و [SEP] اضافه می‌شوند.
    enc = tokenizer(sentence)
    ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("جمله:        ", sentence)
    print("توکن‌ها:      ", tokens)
    print("شناسه‌ها:     ", ids)
    print("تعداد توکن‌ها:", len(ids), "(با توکن‌های ویژه)")
    print("تعداد بدون ویژه:", len(tokenizer(sentence, add_special_tokens=False)["input_ids"]))
    print("-" * 60)
