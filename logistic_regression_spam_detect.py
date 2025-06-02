import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample SMS dataset
data = {
    'label': ['ham', 'spam', 'ham', 'ham', 'spam', 'spam', 'ham'],
    'message': [
        "Hey, how are you?",
        "Congratulations! You've won a prize. Call now!",
        "Let's catch up tomorrow.",
        "Are you coming to the meeting?",
        "Win cash now! Click here!",
        "Get a free ticket now!",
        "Don't forget to bring your notes"
    ]
}

df = pd.DataFrame(data)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Fix: stratify to balance class labels in train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.4, random_state=42, stratify=df['label']
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict & evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
