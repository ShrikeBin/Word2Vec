import re
import numpy as np
import tqdm

class DataLoader:
    def __init__(self, filepaths):
        full_raw_text = ""
        for filepath in filepaths:
            with open(filepath, 'r') as file:
                full_raw_text += file.read()

        self.tokens = self.tokenize(full_raw_text)
        self.vocab = sorted(list(set(self.tokens)))
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def tokenize(self, text):
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        return pattern.findall(text.lower())
    
    def generate_training_data(self, window_size):
        training_data = []
        for i in range(window_size, len(self.tokens) - window_size):
            target_word = self.tokens[i]
            target_index = self.word_to_index[target_word]
            
            context_indices = [self.word_to_index[self.tokens[j]] for j in range(i - window_size, i + window_size + 1) if j != i]
            training_data.append((context_indices, target_index))
        return training_data


SETTINGS = \
{
	'window_size': 2,	        # context window +/- in regard of center word
	'n': 100,		            # dimensions of word embeddings
	'epochs': 5,		        # number of training epochs
	'learning_rate': 0.1	    # learning rate
}

data_loader = DataLoader(['blindsight.txt'])
VOCAB = data_loader.vocab 													# set of unique words in the text
TRAINING_DATA = data_loader.generate_training_data(SETTINGS['window_size']) # list of (context, target) pairs

class Word2Vec:
	'''
		CBOW implementation of Word2Vec. 
		With Negative Sampling. (? not yet)
	'''

	def __init__(self, vocab = VOCAB, settings = SETTINGS):
		self.window_size = settings['window_size']
		self.dim = settings['n']
		self.lr = settings['learning_rate']
		self.epochs = settings['epochs']

		self.vocab = set(vocab)
		self.vocab_size = len(self.vocab)

		self.l1 = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.dim))
		self.l2 = np.random.uniform(-0.5, 0.5, (self.dim, self.vocab_size))

	def forward(self, x):
		hidden = np.dot(self.l1.T, x)
		output = np.dot(self.l2.T, hidden)
		out_prob = self.softmax(output)
		return out_prob
          
	def backward(self, context_indices, hidden, error):
		d_l2 = np.outer(hidden, error)
		self.l2 -= self.lr * d_l2

		hidden_error = np.dot(self.l2, error)
		
		for idx in context_indices:
			self.l1[idx] -= self.lr * (hidden_error / len(context_indices))

	def train(self, training_data):
		for epoch in range(self.epochs):
			loss = 0
			ITER = 0
			for context_indices, target_idx in training_data:
				ITER += 1
				hidden = np.mean(self.l1[context_indices], axis=0)
				u = np.dot(self.l2.T, hidden)
				y_pred = self.softmax(u)

				error = y_pred.copy()
				error[target_idx] -= 1

				self.backward(context_indices, hidden, error)
				
				loss -= np.log(y_pred[target_idx] + 1e-9)

				print(f"Epoch {epoch+1}/{self.epochs} | Iteration {ITER}/{len(training_data)} | Loss: {loss:.4f}", end='\r')
			print(f"Epoch {epoch} complete. Loss: {loss}")

	def softmax(self, x):
		exp_x = np.exp(x - np.max(x))
		return exp_x / exp_x.sum(axis=0)
	

model = Word2Vec()
model.train(TRAINING_DATA)