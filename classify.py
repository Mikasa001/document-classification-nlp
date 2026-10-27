from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# 3. Convert text to numbers (TF-IDF)
vec = TfidfVectorizer(stop_words='english', max_features=20000)
X_train_t = vec.fit_transform(X_train)
X_test_t = vec.transform(X_test)

# 4. Train model
model = MultinomialNB()
model.fit(X_train_t, y_train)

# 5. Test model
pred = model.predict(X_test_t)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=data.target_names))

# 6. Save confusion matrix picture
fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=data.target_names,
    xticks_rotation='vertical', ax=ax, colorbar=False)
plt.tight_layout()
plt.savefig('confusion_matrix.png')