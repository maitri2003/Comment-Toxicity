import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

# Load dataset
df = pd.read_csv("../data/train.csv")


# 👇 Adjust column names based on your CSV
TEXT_COL = "comment_text"   # change if different
LABEL_COL = "toxic"         # change if different

X = df[TEXT_COL].astype(str)
y = df[LABEL_COL]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1,2), max_features=100000)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

# Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/toxicity_model.joblib")
print("✅ Model saved to models/toxicity_model.joblib")
