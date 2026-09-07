from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    text = ""

    if request.method == "POST":
        text = request.form["text"]

        if text.strip():
            # Convert the text into numbers
            text_vector = vectorizer.transform([text])

            # Predict sentiment
            prediction = model.predict(text_vector)

            sentiment = prediction[0]

    return render_template(
        "index.html",
        sentiment=sentiment,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)
