"""مدل پایه: TF-IDF + LogisticRegression، برای مقایسه با فاین‌تیون ParsBERT.

TF-IDF هر خبر را به برداری از وزن کلمه‌ها تبدیل می‌کند: کلمه‌ای که در این خبر زیاد و در بقیهٔ
خبرها کم آمده، وزن بیشتری می‌گیرد. LogisticRegression روی این بردارها یک دسته‌بند خطی می‌سازد.
ترتیب و معنای کلمه‌ها دیده نمی‌شود؛ فقط این‌که کدام کلمه‌ها (و جفت‌کلمه‌ها) آمده‌اند.

اجرا:
    python baseline_tfidf.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
labels = sorted(train["label"].unique())

model = make_pipeline(
    # تک‌کلمه‌ها و جفت‌کلمه‌ها؛ کلمه‌هایی که فقط یک بار آمده‌اند کنار گذاشته می‌شوند.
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=10),
)
model.fit(train["text"], train["label"])
pred = model.predict(val["text"])

print(f"دقت val: {accuracy_score(val['label'], pred):.4f}")
print(f"Macro-F1 val: {f1_score(val['label'], pred, average='macro'):.4f}")
print(classification_report(val["label"], pred, labels=labels, digits=4))
cm = pd.DataFrame(confusion_matrix(val["label"], pred, labels=labels), index=labels, columns=labels)
print("ماتریس درهم‌ریختگی (سطر=واقعی، ستون=پیش‌بینی):")
print(cm.to_string())
