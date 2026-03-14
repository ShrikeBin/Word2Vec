import re

with open('blindsight.txt', 'r') as file:
    text = file.read()

def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

data = tokenize(text)

settings = {
	'window_size': 2,	        # context window +- center word
	'n': 10,		            # dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 50,		        # number of training epochs
	'learning_rate': 0.01	    # learning rate
}