# SMS Spam Detection: Machine Learning vs Deep Learning

A mini project that compares two approaches to detect spam SMS messages:

| | Machine learning | Deep learning |
|---|---|---|
| Model | Linear SVM | Neural network (Embedding + BiLSTM) |
| Text to numbers | TF-IDF | TextVectorization + Embedding layer |
| Library | scikit-learn | TensorFlow / Keras |

Both models are trained and tested on exactly the same data, then compared
with accuracy and F1-score on the spam class. A Streamlit web app lets anyone
type an SMS and see the answer of both models.

## Dataset

SMS Spam Collection: about 5,500 English SMS messages labelled `ham` (normal)
or `spam`. After removing duplicates, 80% is used for training and 20% for
testing (`random_state=42`, same split for both models).

## Project structure

```
SMS-Spam-ML-vs-DL/
├── app/
│   └── main.py              Streamlit web app
├── data.py                  Loads the data and makes the train/test split
├── train_ml.py              Trains the SVM (machine learning)
├── train_dl.py              Trains the neural network (deep learning)
├── compare.py               Compares both models and draws a chart
├── spam_data.csv            Dataset
├── requirements.txt         Python libraries
└── README.md
```

Files created by the scripts: `svm_model.pkl`, `tfidf_vectorizer.pkl`,
`dl_model.keras`, `results_ml.json`, `results_dl.json`, `comparison.png`.

## How to run

Tested on macOS (Apple Silicon) with Python 3.11 and TensorFlow 2.21. A clean
Python 3.11 environment is recommended, because TensorFlow can crash inside a
large shared environment such as the Anaconda base environment.

```bash
conda create -n spamdl python=3.11 -y
conda activate spamdl
pip3 install -r requirements.txt
python3 train_ml.py      # trains the SVM
python3 train_dl.py      # trains the neural network
python3 compare.py       # prints the comparison and saves comparison.png
streamlit run app/main.py
```

## Results (test set, 1,034 messages)

| Model | Accuracy | F1-score (spam) | Precision (spam) | Recall (spam) | Training time |
|---|---|---|---|---|---|
| SVM (machine learning) | 0.9797 | 0.9143 | 0.98 | 0.85 | 0.01 s |
| Neural network (deep learning) | 0.9797 | 0.9150 | 0.97 | 0.86 | 4.68 s |

Both models classify normal messages almost perfectly (recall 1.00 on `ham`).
The difficult part is spam: about 14 to 15% of the real spam messages are
missed by both models, while 97 to 98% of the messages they call spam really
are spam.

## Conclusion

On this dataset, machine learning and deep learning reach the same accuracy
(0.9797). The neural network has a slightly higher F1-score on the spam class
(0.9150 against 0.9143), but this difference is tiny. The test set contains
only 131 spam messages, so one message more or less changes the score by
more than that. We cannot say that one model is really better.

The real difference is the cost. The SVM trains in 0.01 seconds with a single
library, while the neural network needs 4.68 seconds, TensorFlow and a more
complex setup. For short SMS texts and a small dataset, a simple model with
good features (TF-IDF + SVM) is a strong baseline. Deep learning would
probably become more useful with a much larger dataset or with longer and more
varied texts. These results come from one run with one random split, so they
should be confirmed with cross-validation before drawing a final conclusion.

## Author

Kouassi Kouame Daniel
