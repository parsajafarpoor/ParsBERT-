# فاین‌تیون ParsBERT برای دسته‌بندی اخبار فارسی

دسته‌بندی خلاصهٔ خبرهای فارسی در ۶ کلاس: اقتصاد، بین‌الملل، ورزش، علم و فناوری، سیاست، سلامت.
مدل پایه: [`HooshvareLab/bert-fa-zwnj-base`](https://huggingface.co/HooshvareLab/bert-fa-zwnj-base)

گزارش کامل کار: [`docs/finetuning-report.pdf`](docs/finetuning-report.pdf)

## نتیجه‌ها (روی val.csv، ۶۰۰ خبر)
| مدل | دقت | Macro-F1 |
|---|---|---|
| TF-IDF + LogisticRegression (`baseline_tfidf.py`) | 0.8467 | 0.8463 |
| ParsBERT با BERT منجمد، فقط classifier (`--freeze_bert --lr 1e-3 --epochs 5`) | 0.7717 | 0.7709 |
| **ParsBERT فاین‌تیون کامل (مدل نهایی، `--output_dir outputs_report`)** | **0.8883** | **0.8878** |

- اجراهای مختلف فاین‌تیون کامل (seed و سخت‌افزار متفاوت) بین ۸۷٫۵٪ و ۸۹٫۷٪ دقت داشتند.
- بیشترین اشتباه بین «سیاست» و «بین‌الملل» است؛ بخش زیادی از آن ناشی از ناسازگاری برچسب‌ها در داده است.
- ترفندهای `balance.py` و `knn_fix.py` روی val کمکی نکردند.

## ساختار
```
data/                  train.csv (۳۰۰۰)، val.csv (۶۰۰)، test.csv (۱۲۰۰ بدون برچسب)
train.py               آموزش، ارزیابی، ماتریس درهم‌ریختگی، جدول دوره‌ها، نمودار یادگیری، پیش‌بینی test
part1_tokenizer.py     بررسی توکنایزر و توزیع طول توکن‌ها
part4_analysis.py      احتمال‌های softmax برای جمله‌های مرزی و تحلیل اشتباه‌های val
baseline_tfidf.py      مدل پایهٔ TF-IDF + LogisticRegression
ensemble.py            ترکیب (میانگین احتمال) چند اجرا
tapt.py                ادامهٔ پیش‌آموزش MLM روی متن خبرها (TAPT)
find_label_errors.py   یافتن برچسب‌های احتمالاً اشتباه با k-fold (confident learning)
balance.py, knn_fix.py ترفندهای پس‌پردازش (روی این داده کمک نکردند)
tokenizer_demo.py      نمونهٔ سادهٔ استفاده از توکنایزر
docs/                  گزارش PDF
```
هر اجرای `train.py` خروجی‌هایش را در پوشهٔ `--output_dir` می‌گذارد: `predictions.csv`، `val_errors.csv`،
`epochs.csv`، `learning_curve.png`، `val_logits.npy`، `test_logits.npy` و `best_model/`.

## اجرا روی Google Colab
1. در Colab از منوی **Runtime → Change runtime type** گزینهٔ **T4 GPU** را انتخاب کنید.
2. این سلول‌ها را اجرا کنید:
```python
!git clone -b claude/loving-einstein-jh2u6g https://github.com/parsajafarpoor/ParsBERT-.git
%cd ParsBERT-
!pip install -q -r requirements.txt
!python train.py --output_dir outputs_report
```
3. فایل نتیجه را دانلود کنید:
```python
from google.colab import files
files.download("outputs_report/predictions.csv")
```
روی T4 آموزش حدود ۳ دقیقه طول می‌کشد.

## تنظیمات قابل تغییر
| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `--model` | `HooshvareLab/bert-fa-zwnj-base` | مدل پایه (یا پوشهٔ خروجی `tapt.py`) |
| `--epochs` | 3 | تعداد دور آموزش روی کل داده |
| `--lr` | 2e-5 | نرخ یادگیری |
| `--batch_size` | 16 | تعداد نمونه در هر گام |
| `--max_length` | 128 | حداکثر طول متن به توکن |
| `--seed` | 42 | عدد شروع تصادفی‌ها |
| `--output_dir` | outputs | پوشهٔ خروجی‌های این اجرا |
| `--no_fp16` | خاموش | خاموش کردن fp16 (اگر loss روی GPU برابر nan شد) |
| `--freeze_bert` | خاموش | فقط classifier آموزش ببیند |
| `--train_on_all` | خاموش | آموزش روی train+val (بدون ارزیابی) |
| `--label_fixes` | - | اعمال اصلاح برچسب‌ها از خروجی `find_label_errors.py` |

## ابزارهای دیگر
- `python ensemble.py outputs_s1 outputs_s2 ...`: میانگین احتمال‌های چند اجرا؛ دقت val هر مدل و ترکیب را
  چاپ می‌کند و `predictions_ensemble.csv` می‌سازد.
- `python tapt.py` و بعد `python train.py --model outputs/tapt_model`: آشنا کردن مدل با متن خبرها پیش از دسته‌بندی.
- `python find_label_errors.py`: برچسب‌های مشکوک train را در `suspect_labels.csv` می‌نویسد. ستون `new_label`
  را پر کنید (برچسب درست، یا «حذف») و بعد `python train.py --label_fixes suspect_labels.csv`.
- `python balance.py` و `python knn_fix.py`: پس‌پردازش پیش‌بینی‌ها؛ هر دو اول اثرشان را روی val نشان می‌دهند.

## منبع داده
زیرمجموعه‌ای از [PN-Summary](https://github.com/hooshvare/pn-summary) (Hooshvare Team, MIT License). جزئیات در `data/README.txt`.
