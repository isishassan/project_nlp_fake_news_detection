# Fake or Real? Detecting News with NLP

Group project with [Eya Cherif](https://github.com/echerif18) and [Suzana Cracco](https://github.com/suzanacracco-max), created during the Ironhack Data Science & Machine Learning bootcamp, February 2026

## 📌 Overview

Can machines detect fake news? This project builds and compares multiple NLP classification models to label news as real (1) or fake (0). Models were trained on the Kaggle Fake News dataset (32,206 articles) and evaluated on a custom 2025 headline test set (Onion headlines vs real headlines) to measure real-world generalization. The focus was not just validation accuracy — but robustness on unseen data.



## 🎯 Objective

- Compare NLP vectorization techniques
- Evaluate multiple classification models
- Measure the generalization gap
- Identify which approaches transfer best to new data

## 📂 Data
Training Data
- Kaggle Fake News Detection dataset
- 32,206 labeled articles
- Balanced classes (Fake / Real)

Custom Test Set (2025 Headlines)
- 100 headlines total
- 50 real (CNN, Wikipedia, Britannica, CBS)
- 50 satirical (The Onion)

Designed to test distribution shift and model robustness.

## ⚙️ Pipeline
1️⃣ Preprocessing
- Removed duplicates
- Cleaned punctuation & lowercased text
- Stopword removal
- Optional lemmatization
- Added engineered feature: contains_swearword

2️⃣ Vectorization
- Bag of Words (CountVectorizer)
- TF-IDF
- N-grams (1–2 grams)
- Feature & max_df tuning

3️⃣ Models Evaluated
- Logistic Regression
- XGBoost
- Multinomial Naive Bayes
- BERT
- DistilBERT

4️⃣ Hyperparameter Tuning
- GridSearchCV
- RandomizedSearchCV
- Sklearn Pipeline

### Best configuration:

```CountVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    max_df=0.879
)

LogisticRegression(
    C=7.76,
    penalty="l2",
    solver="lbfgs"
)
```

## 📊 Results
Validation (80/20 Split): Most models achieved 90–98% accuracy.

### Examples:
- BoW + Logistic Regression → ~0.94
- Naive Bayes (TF-IDF) → ~0.98
- BERT → ~0.98

### 🚨 Custom 2025 Test Performance
Model	Test Accuracy
- BoW + Logistic Regression (Tuned)	0.73
- TF-IDF + Logistic Regression	0.69
- BERT + Swear Flag	0.66
- TF-IDF + XGBoost (Lemmatized)	0.64
- Naive Bayes	~0.30

Best Model: Tuned Logistic Regression (BoW, 1–2 grams)
→ 73% accuracy on unseen headlines

## 🧠 Key Insights

- Simplicity Wins: Logistic Regression outperformed more complex models.
- Validation ≠ Real-World Performance
- Accuracy dropped significantly on new 2025 data.
- Preprocessing Impact Varies: Lemmatization helped XGBoost but was less critical for linear models.
- Transformers Show Potential: Strong validation performance but still sensitive to distribution shift.

## 🚀 Future Work
- Larger, more diverse datasets
- Domain adaptation
- Data augmentation
- Improved satire detection
- Transformer fine-tuning on recent data

## 🏁 Final Takeaway

High validation accuracy does not guarantee real-world performance. In this project, a well-tuned Logistic Regression model outperformed more complex architectures on unseen 2025 headlines. 

Sometimes, simple — done well — wins.
