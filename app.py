from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)

@app.route("/")
def index():
    return "Question Answering System"

@app.route("/question", methods=['POST'])
def question():
    # Load the pre-trained question answering model
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    text = request.form.get('text')
    question = request.form.get('question')

    # Perform question answering
    result = qa_model(question=question, context=text)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
