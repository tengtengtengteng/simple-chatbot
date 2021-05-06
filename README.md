# Simple Chatbot

This simple project was completed by following this [tutorial](https://www.youtube.com/watch?v=1lwddP0KUEg) by [NeuralNine](https://www.youtube.com/channel/UC8wZnXYK_CGKlBcZp-GxYPA). 

A simple neural network was constructed using ```TensorFlow``` and trained on the phrases and classes in the intents.json file. The main idea is to convert all the words in the json file into a bag of words and classify each phrase based on the words that are present. Based on the predicted class, the chatbot will randomly pick a response that was defined in the json file to reply the user.

To customise the chatbot, the classes and phrases in the json file can be updated to suit a particular function (need to think about what our users will try to ask our chatbot) before running training.py to train a new model. The trained model will be saved into a h5 file, which can be loaded when deploying the chatbot.
