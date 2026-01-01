# Spam Email Detection using Machine Learning 

##  Overview

This project implements a **Spam Email Detector** using Natural Language Processing (NLP) and supervised machine learning. The model classifies emails as **Spam** or **Not Spam (Ham)** based on their content. It combines text preprocessing, feature extraction, and classification to build an accurate spam filter.

---

##  Objective

To develop a model that detects whether an email is spam or not by analyzing its content using NLP and machine learning algorithms.

---

##  Problem Type

- **Type**: Supervised Learning
- **Task**: Binary Classification
- **Labels**: Spam (1), Not Spam (0)
- **Domain**: NLP / Email Filtering

---

##  Tech Stack

| Component       | Technology               |
|-----------------|--------------------------|
| Language        | Python                   |
| Libraries       | `nltk`, `scikit-learn`, `pandas`, `matplotlib` |
| Feature Extraction | Bag of Words / TF-IDF |
| Classifier      | Naive Bayes / Logistic Regression / SVM |

---

---

##  Dataset

- Source: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Columns:
  - `label`: `spam` or `ham`
  - `text`: email or SMS content
- Dataset is preprocessed to clean and label the entries numerically (`1` = spam, `0` = ham)

---

##  NLP Preprocessing Steps

1. Lowercasing
2. Removing punctuation, numbers, and special characters
3. Tokenization
4. Stopword removal
5. Stemming (PorterStemmer)
6. Feature vectorization using **Bag of Words** or **TF-IDF**

---

##  How to Run the Project

###  Clone the Repository

```bash
git clone https://github.com/adnanabdullah0405/spam-email-classifier.git
cd spam-email-classifier


