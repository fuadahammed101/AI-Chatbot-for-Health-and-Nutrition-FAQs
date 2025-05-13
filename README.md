# AI-Chatbot-for-Health-and-Nutrition-FAQs


This is a simple command-line chatbot that provides answers to common health and nutrition questions. It uses TF-IDF vectorization and cosine similarity to find the most relevant answer from a predefined FAQ dataset.

## Features

- Answers questions related to diet, hydration, exercise, and nutrition.
- Uses natural language processing techniques to match user queries with FAQ questions.
- Provides scientifically backed nutrition advice.

## Prerequisites

- Python 3.6 or higher
- `scikit-learn` library

## Installation

1. Clone or download the project files.

2. Install the required Python package:

```bash
pip install scikit-learn
```

## Usage

Run the chatbot script from the command line:

```bash
python chatbot.py
```

You will see a welcome message. Type your questions about health and nutrition and press Enter. Type `exit` to quit the chatbot.

Example:

```
You: What are the best foods for weight loss?
Chatbot: Foods rich in fiber, protein, and healthy fats like vegetables, lean meats, and nuts are good for weight loss.
```

## How It Works

- The chatbot preprocesses the FAQ questions by lowercasing and removing punctuation.
- It uses `TfidfVectorizer` from scikit-learn to convert questions into TF-IDF vectors.
- When a user inputs a query, it is preprocessed and transformed into a TF-IDF vector.
- Cosine similarity is calculated between the user query vector and all FAQ question vectors.
- The answer corresponding to the most similar question is returned if the similarity score is above a threshold; otherwise, a default message is shown.

## License

This project is for educational purposes.
