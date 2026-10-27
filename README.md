# Document Classification using NLP

Classifies news documents into 20 categories using TF-IDF and a Naive Bayes classifier.

## Dataset
20 Newsgroups (about 18,800 documents, 20 categories), loaded from scikit-learn.

## Method
1. Load the dataset (headers, footers and quotes removed)
2. Split into 80% training and 20% testing
3. Convert text to numbers using TF-IDF
4. Train a Multinomial Naive Bayes model
5. Evaluate with accuracy, a classification report and a confusion matrix

## Result
Accuracy: 72% on 3,770 test documents.

## How to run
pip install -r requirements.txt
python classify.py

## Files
- classify.py : main code
- results.txt : full evaluation output
- confusion_matrix.png : confusion matrix