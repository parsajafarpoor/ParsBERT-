# فاین‌تیون ParsBERT برای دسته‌بندی اخبار فارسی

دسته‌بندی خلاصهٔ خبرهای فارسی در ۶ کلاس: اقتصاد، بین‌الملل، ورزش، علم و فناوری، سیاست، سلامت.
مدل پایه: [`HooshvareLab/bert-fa-zwnj-base`](https://huggingface.co/HooshvareLab/bert-fa-zwnj-base)

## ساختار
```
data/train.csv      ۳۰۰۰ خبر با برچسب (۵۰۰ از هر کلاس)
data/val.csv        ۶۰۰ خبر با برچسب برای ارزیابی حین آموزش
data/test.csv       ۱۲۰۰ خبر بدون برچسب (id,text)
train.py            اسکریپت آموزش و پیش‌بینی
```

## اجرا روی Google Colab
1. در Colab از منوی **Runtime → Change runtime type** گزینهٔ **T4 GPU** را انتخاب کنید.
2. این سلول‌ها را اجرا کنید:
```python
!git clone -b claude/loving-einstein-jh2u6g https://github.com/parsajafarpoor/ParsBERT-.git
%cd ParsBERT-
!pip install -q -r requirements.txt
!python train.py
```
3. فایل نتیجه را دانلود کنید:
```python
from google.colab import files
files.download("predictions.csv")
```
روی T4 آموزش حدود ۵ دقیقه طول می‌کشد.

## تنظیمات قابل تغییر
| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `--model` | `HooshvareLab/bert-fa-zwnj-base` | مدل پایه |
| `--epochs` | 3 | تعداد دور آموزش روی کل داده |
| `--lr` | 2e-5 | نرخ یادگیری |
| `--batch_size` | 16 | تعداد نمونه در هر گام |
| `--max_length` | 128 | حداکثر طول متن به توکن |

مثال: `!python train.py --epochs 4 --lr 3e-5`

## بهتر کردن دقت
- آخر هر اجرا ماتریس درهم‌ریختگی چاپ می‌شود و خبرهای اشتباه در `val_errors.csv` ذخیره می‌شوند.
- `python balance.py`: اگر مدل یک کلاس را بیش از حد پیش‌بینی کند (مثلاً «بین‌الملل» به‌جای «سیاست»)،
  با فرض متعادل بودن داده، پیش‌بینی‌های مرزی را جابه‌جا می‌کند و `predictions_balanced.csv` می‌سازد.
  اول دقت val قبل و بعد را چاپ می‌کند؛ فقط اگر بهتر شد از آن استفاده کنید.
- `python knn_fix.py`: اگر خبری از test تقریباً همان متن یک خبر train باشد، برچسب آن را می‌گیرد.
  آستانهٔ شباهت را روی val انتخاب می‌کند و فقط اگر کمک کرد `predictions_knn.csv` می‌سازد.
- `python train.py --train_on_all`: آموزش نهایی روی train+val، بعد از پیدا کردن تنظیمات خوب.

## منبع داده
زیرمجموعه‌ای از [PN-Summary](https://github.com/hooshvare/pn-summary) (Hooshvare Team, MIT License). جزئیات در `data/README.txt`.
