Document Classification Using NLP Techniques

Project Overview

This project implements an automated document classification system using Natural Language Processing (NLP) and machine learning. It classifies text documents into predefined categories using the 20 Newsgroups dataset.

Objectives

- Process and analyse text documents using NLP techniques.
- Convert text into numerical features using TF-IDF.
- Train a machine learning model to classify documents.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Visualise classification performance using a confusion matrix.

Technologies Used

- Python
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Matplotlib

Dataset

The project uses the 20 Newsgroups dataset provided by Scikit-learn. It contains text documents belonging to 20 different newsgroup categories.

The dataset is divided into:

- 80% training data
- 20% testing data

Headers, footers, and quoted text are removed when loading the dataset.

Methodology

1. Load the 20 Newsgroups dataset.
2. Split the dataset into training and testing sets.
3. Convert text into numerical features using TF-IDF.
4. Train a Multinomial Naive Bayes classifier.
5. Predict categories for the test documents.
6. Evaluate the predictions using classification metrics.
7. Generate and save a confusion matrix.

Model Performance

The recorded model evaluation results are:

- Accuracy: 72.18%
- Macro-average F1-score: 0.70
- Weighted-average F1-score: 0.71
- Test documents: 3,770

These results provide a baseline evaluation of the model on the selected dataset. Performance varies across categories.

Project Files

- "classify.py" — Python source code for document classification.
- "requirements.txt" — Required Python libraries.
- "results.txt" — Model evaluation output.
- "confusion_matrix.png" — Visualisation of classification results.
- "README.md" — Project documentation.

Installation and Execution

1. Clone the repository

git clone https://github.com/Mikasa001/document-classification-nlp.git
cd document-classification-nlp

2. Install dependencies

pip install -r requirements.txt

3. Run the project

python classify.py

The script prints the model's accuracy and classification report and saves the confusion matrix as "confusion_matrix.png".

Limitations

- The model uses a traditional machine learning algorithm.
- Classification performance varies between categories.
- The system is designed for the dataset's predefined categories rather than arbitrary document types.

Future Improvements

- Compare Naive Bayes with Logistic Regression and Support Vector Machines.
- Experiment with improved text preprocessing.
- Evaluate alternative feature extraction techniques.
- Develop a simple web interface for classifying user-provided documents.

Conclusion

This project demonstrates how NLP and machine learning can be combined to automate document classification. TF-IDF represents the text numerically, while Multinomial Naive Bayes predicts document categories. Evaluation metrics and a confusion matrix help assess the system's performance.