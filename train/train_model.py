import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("dataset.csv")

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vect, y)

with open("../model/email_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
