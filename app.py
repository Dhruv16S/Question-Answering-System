from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)

@app.route("/")
def index():
    return "Question Answering System"

@app.route("/question", methods=['GET', 'POST'])
def question():
    qa_model = pipeline("question-answering")
    text = request.form.get('text')
    question = request.form.get('question')
    result = qa_model(question=question, context=text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True)