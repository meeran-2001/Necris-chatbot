
import streamlit as st
st.title('NecRis Chatbot')
st.write("Hello from NecRis chatbot!")
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
from tensorflow.keras.models import load_model

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('intents.json') as f:
    intents = json.load(f)

# Load trained model (make sure chatbot_model.h5 is in repo)
model = load_model('chatbot_model.h5')

lemmatizer = WordNetLemmatizer()

# Prepare words and classes (you must generate and save these lists during training and add them here)
words = [  # example, replace with your actual words list from training
    'hi', 'hello', 'anxious', 'feeling', 'nervous', 'calm', 'mind', 'racing'
]
classes = ['greeting', 'anxiety']  # replace with your actual classes list

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't get that. Could you please rephrase?"
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, something went wrong."

st.title("Necris - Your Mental Health Chatbot 💬")

user_input = st.text_input("You:")

if user_input:
    ints = predict_class(user_input)
    response = get_response(ints, intents)
    st.text_area("Necris:", value=response, height=150)

