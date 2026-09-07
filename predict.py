import pickle

# Load the trained model
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

print("Sentiment Analysis System")
print("-------------------------")

while True:
    text = input("Enter a sentence (or type 'exit' to stop): ")

    if text.lower() == "exit":
        print("Thank you for using the Sentiment Analysis System!")
        break

    # Convert the input text into numbers
    text_vector = vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(text_vector)

    print("Predicted Sentiment:", prediction[0])
    print()
