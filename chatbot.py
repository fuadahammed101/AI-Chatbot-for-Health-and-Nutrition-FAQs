import json
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('data.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def get_answer(question):
    questions = [preprocess_text(item["question"]) for item in faq_data]
    answers = [item["answer"] for item in faq_data]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)

    user_query = preprocess_text(question)
    user_vec = vectorizer.transform([user_query])

    similarities = cosine_similarity(user_vec, tfidf_matrix)
    max_sim_index = similarities.argmax()
    max_sim_score = similarities[0, max_sim_index]

    if max_sim_score < 0.1:
        return "Sorry, I don't have an answer for that. Please try asking something else."
    else:
        return answers[max_sim_index]

if __name__ == "__main__":
    print("Welcome to the Health & Nutrition Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = get_answer(user_input)
        print("Chatbot:", response)
