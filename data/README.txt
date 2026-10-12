Persian news, 6 classes (a subset of PN-Summary)
================================================

text   the summary of a Persian news article (one or two sentences)
label  its category: اقتصاد، بین‌الملل، ورزش، علم و فناوری، سیاست، سلامت

train.csv  3000 rows (500 per class), with labels
val.csv     600 rows (100 per class), with labels
test.csv   1200 rows, columns id and text, WITHOUT labels. Predict them and upload predictions.csv
           with two columns: id,label (label written exactly as in train.csv).

Source: PN-Summary, Farahani et al., "Leveraging ParsBERT and Pretrained mT5 for Persian Abstractive Text
Summarization" (2020), https://github.com/hooshvare/pn-summary, released under the MIT License:

Copyright (c) 2020 Hooshvare Team
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Subset (balanced sampling, de-duplicated) made for the shokrzad.com/ai course.
