from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Training data
texts = [
    "I love this product",
    "This is amazing",
    "I am very happy",
    "Excellent experience",
    "I really enjoyed it",
    "This is wonderful",
    "I hate this product",
    "This is terrible",
    "I am very disappointed",
    "Worst experience",
    "I don't like it",
    "This is horrible",
    "The product is okay",
    "It is average",
    "Nothing special",
    "It was fine",
    "The experience was normal",
    "It is not bad"
]

labels = [
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral"
]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train the machine learning model
model = MultinomialNB()
model.fit(X, labels)

# Save the trained model
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Sentiment model trained successfully!")
