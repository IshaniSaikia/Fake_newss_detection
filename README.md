# Fake_newss_detection
# Fake News Detection Using NLP

## Overview
A machine learning system that classifies news articles as **Fake** or **Real** using natural language processing and TF-IDF feature extraction. Built as a hands-on NLP/ML project comparing two classic text classification approaches.

## Problem Statement
Misinformation spreads easily through news articles that mimic legitimate journalism. This project explores whether a text classifier can learn to distinguish fake from real news articles based on writing patterns in a labeled dataset.

## Objective
- Build a text classification pipeline from raw news articles to prediction
- Compare Logistic Regression and Multinomial Naive Bayes on the same features
- Evaluate honestly using standard classification metrics

## Dataset
[Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — ~23.5K fake and ~21.4K real news articles, each with `title`, `text`, `subject`, and `date`.

## Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK

## Methodology

### Data Preprocessing
- Combined `title` + `text` into a single input field
- Removed exact duplicate articles and broken/URL-only entries
- Lowercased text, removed URLs, punctuation, and numbers
- Removed English stopwords (NLTK)
- Deliberately excluded the `subject` column as a feature, since it perfectly separated Fake vs. Real sources in this dataset and would have caused data leakage rather than genuine learning

### Feature Extraction — TF-IDF
Used `TfidfVectorizer` (max 5,000 features, unigrams) fitted only on the training set to prevent data leakage into the test set.

### Models
- **Logistic Regression** — linear classifier well-suited to high-dimensional sparse text data
- **Multinomial Naive Bayes** — probabilistic classifier commonly used as a text classification baseline

## Evaluation Metrics
Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Actual Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 98.75% | 98.34% | 99.36% | 98.85% |
| Multinomial Naive Bayes | 93.90% | 94.38% | 94.36% | 94.37% |

Logistic Regression outperformed Naive Bayes across every metric in this experiment.

## Error Analysis
Misclassifications were examined directly. Fake articles written in a neutral, wire-service tone were sometimes predicted as Real, and Real feature/human-interest articles with narrative (rather than terse political wire) style were sometimes predicted as Fake. This suggests the model primarily learns **stylistic and tonal patterns** rather than verifying factual content — an important limitation, not a flaw specific to this implementation.

## Example Prediction
A custom `predict_news(text)` function allows classification of new text:
```python
predict_news("Senate committee to review new infrastructure spending bill next week")
# Prediction: Real | Confidence: 81.81%
```

## Limitations
- This is a pattern-based classifier, not a fact-checking system — it cannot verify whether a claim is true
- Performance is strongest on political news (the dominant topic in this dataset) and less confident on unfamiliar topics or neutral tones
- Fake and Real articles in this dataset originate from largely distinct sources, which may make the classification task easier than distinguishing misinformation in the wild

## Future Improvements
- Test on a more diverse, out-of-domain dataset to check generalization
- Experiment with n-grams or word embeddings instead of single-word TF-IDF
- Try additional models (SVM, Random Forest) for comparison

## How to Run
1. Clone this repository
2. Download the dataset from the Kaggle link above and place `Fake.csv`/`True.csv` in the `data/` folder
3. Install dependencies: `pip install -r requirements.txt`
4. Open and run `fake_news_detection.ipynb` in Jupyter or Google Colab

## Dataset Source
[Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon

## Author
Isha
