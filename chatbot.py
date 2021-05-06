import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# tokenize and lemmatize sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# create bag of words for sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# predict and return the one with largest probability
def predict_class(sentence):
    bow = bag_of_words(sentence) # transform sentence into bag of words
    res = model.predict(np.array([bow]))[0] # make prediction
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] # get index and probability if above threshold
    results.sort(key=lambda x: x[1], reverse=True) # sort in descending order based on probability
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# based on predicted class, randomly return a response
def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    except IndexError: # no probability above threshold. model can't figure out what class
        result = "Sorry I don't understand!"
    return result

if __name__ == "__main__":
    print("ChatBot is running! Type'quit' to leave.")
    while True:
        message = input("")
        if message.lower() == 'quit':
            break
        else:
            ints = predict_class(message)
            res = get_response(ints, intents)
            print(res)