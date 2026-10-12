"""بخش ۴ تکلیف: پیش‌بینی جمله‌های مرزی و تحلیل اشتباه‌های val با مدل نهایی.

اجرا (بعد از python train.py --output_dir outputs_report):
    python part4_analysis.py --model_dir outputs_report/best_model
"""

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SENTENCES = {
    "الف": "باشگاه استقلال برای جبران زیان مالی، بخشی از سهام خود را در بورس عرضه می‌کند.",
    "ب": "وزیر بهداشت در سفر به ژنو با مدیرکل سازمان جهانی بهداشت دربارهٔ واکسن کرونا گفتگو کرد.",
    "ج": "مجلس طرح حمایت از شرکت‌های دانش‌بنیان در حوزهٔ هوش مصنوعی را تصویب کرد.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="outputs_report/best_model")
    p.add_argument("--n_errors", type=int, default=15)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
    id2label = model.config.id2label

    @torch.no_grad()
    def probs(texts):
        enc = tokenizer(texts, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
        return torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

    # ---------- ۴٫۱ ----------
    print("========== ۴٫۱ جمله‌های مرزی ==========")
    P = probs(list(SENTENCES.values()))
    for (key, text), pr in zip(SENTENCES.items(), P):
        order = pr.argsort()[::-1]
        print(f"{key}) {text}")
        print(f"   پیش‌بینی: {id2label[order[0]]}  احتمال: {pr[order[0]]:.3f}")
        print("   همهٔ احتمال‌ها: " + "، ".join(f"{id2label[i]} {pr[i]:.3f}" for i in order))
        print()

    # ---------- ۴٫۲ ----------
    print("========== ۴٫۲ اشتباه‌های val ==========")
    val = pd.read_csv("data/val.csv")
    P = np.concatenate([probs(val["text"][i:i + 64].tolist()) for i in range(0, len(val), 64)])
    val["predicted"] = [id2label[i] for i in P.argmax(1)]
    val["p_predicted"] = P.max(1).round(3)
    label2id = {v: k for k, v in id2label.items()}
    val["p_true"] = P[np.arange(len(val)), val["label"].map(label2id)].round(3)
    wrong = val[val["label"] != val["predicted"]].sort_values("p_predicted", ascending=False)
    print(f"تعداد اشتباه: {len(wrong)} از {len(val)}")
    print("مرتب از «مطمئن‌ترین اشتباه» (احتمالاً برچسب نادرست) تا «مرددترین» (احتمالاً ابهام):\n")
    for _, r in wrong.head(args.n_errors).iterrows():
        print(f"واقعی: {r['label']}  |  پیش‌بینی: {r['predicted']} ({r['p_predicted']})  |  احتمال برچسب واقعی: {r['p_true']}")
        print(f"   {r['text']}\n")
    wrong.to_csv("val_errors_with_probs.csv", index=False, encoding="utf-8-sig")
    print("همهٔ اشتباه‌ها با احتمال در val_errors_with_probs.csv ذخیره شد.")


if __name__ == "__main__":
    main()
