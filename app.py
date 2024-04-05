from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import os

app = Flask(__name__)

model_name = "all-MiniLM-L6-v2"
model_path = os.path.join("models", model_name)

if not os.path.exists(model_path):
    model = SentenceTransformer(model_name)
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)
else:
    model = SentenceTransformer(model_path)

questions = [
    "What is Python?",
    "How do I install Python?",
    "What are the data types in Python?",
    "How do I create a list in Python?",
    "What is machine learning?",
    "How do I train a machine learning model?"
]

questions_embeddings = model.encode(questions)


@app.route('/similar_question', methods=['POST'])
def get_similar_question():
    data = request.get_json()
    input_question = data['question']
    input_question_embedding = model.encode(input_question)
    similar_questions = util.semantic_search(input_question_embedding, questions_embeddings, top_k=1)
    similar_question_index = similar_questions[0][0]['corpus_id']
    selected_question = questions[similar_question_index]
    return jsonify({"similar_question": selected_question})


if __name__ == '__main__':
    app.run(debug=True)
