# Sentiment Analysis System

An AI-based sentiment analysis system that uses Natural Language Processing (NLP) and Machine Learning to classify text as Positive, Negative, or Neutral.

## Project Overview

This project analyzes the sentiment of user-entered text and predicts whether the text expresses a positive, negative, or neutral sentiment.

The system uses a Machine Learning model and provides a simple web interface using Flask.

## Technologies Used

- Python
- NLP
- Machine Learning
- Scikit-learn
- Flask
- CountVectorizer
- Multinomial Naive Bayes
- HTML & CSS

## How It Works

1. User enters text through the Flask web interface.
2. CountVectorizer converts the text into numerical features.
3. The Multinomial Naive Bayes model analyzes the text.
4. The model predicts the sentiment.
5. The result is displayed as Positive, Negative, or Neutral.

## Project Structure

sentiment-analysis-system/
├── app.py
├── predict.py
├── train_model.py
├── sentiment_model.pkl
├── vectorizer.pkl
└── templates/
    └── index.html

## How to Run

### Install Required Libraries

python -m pip install flask scikit-learn nltk

### Run the Application

python app.py

### Open in Browser

http://127.0.0.1:5000

## Features

- Positive sentiment detection
- Negative sentiment detection
- Neutral sentiment detection
- Simple web interface
- Machine Learning based prediction
- Real-time text analysis

## Future Improvements

- Train the model using a larger dataset
- Add prediction confidence scores
- Improve text preprocessing
- Deploy the application online
- Add advanced NLP techniques

## Author

Uma Mageshwari S
