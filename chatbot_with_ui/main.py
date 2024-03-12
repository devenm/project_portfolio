from flask import Flask, request, jsonify, render_template, redirect, url_for
import openai
import os
import string
import spacy
import csv
import datetime

#TODO: Add tokenization counting

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')
CSV_FILE = 'chatbot_data.csv'

def preprocess_input(input_text):
    tokens = input_text.split()
    tokens = [token.lower() for token in tokens]
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    preprocessed_input = " ".join(lemmatized_tokens)
    return preprocessed_input

def generate_response(input_text):
    client = openai.OpenAI(api_key=os.environ.get('chatbot_oai_key'))
    stream = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[{"role": "user", "content": input_text}],
        stream=True
    )
    response = ""
    for response_data in stream:
        for choice in response_data.choices:
            if choice.delta.content:
                response += choice.delta.content
    return response.strip()

#for data collection and fine tuning
def save_to_csv(input_text, processed_input, input_tokens, response_text, response_tokens, total_tokens):
    with open(CSV_FILE, 'a', newline='') as csvfile:
        fieldnames = ['Input', 'Processed Input', 'Input Tokens', 'Response', 'Response Tokens', 'Total Tokens','Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.stat(CSV_FILE).st_size == 0:
            writer.writeheader()
        date = datetime.date.today()
        writer.writerow({'Input': input_text,
                         'Processed Input': processed_input,
                         'Input Tokens': input_tokens,
                         'Response': response_text,
                         'Response Tokens': response_tokens,
                         'Total Tokens': total_tokens,
                         "Date":date})

# App Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/query_page', methods=['GET', 'POST'])
def query_page():
    responses = []
    error = None
    if request.method == 'POST':
        user_input = request.form['query']
        if not user_input:
            error = 'Please enter a query.'
        else:
            try:
                processed_input = preprocess_input(user_input)
                response = generate_response(processed_input)
                responses.append({'query': user_input, 'response': response})
                save_to_csv(user_input, processed_input, 0, response, 0, 0)
            except Exception as e:
                error = 'An error occurred while processing the query.'
                print(f'Error: {e}')
    return render_template('query_page.html', responses=responses, error=error)

if __name__ == '__main__':
    app.run(debug=True)