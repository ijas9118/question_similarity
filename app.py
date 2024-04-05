from sentence_transformers import SentenceTransformer, util
import os

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
input_question_embedding = model.encode("How do I learn Python?")

print(util.semantic_search(questions_embeddings, input_question_embedding))
