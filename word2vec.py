import re
import numpy as np
from time import sleep


class DataLoader:
    def __init__(self, filepaths):
        full_raw_text = ""
        for filepath in filepaths:
            with open(filepath, "r") as file:
                full_raw_text += file.read()

        self.tokens = self.tokenize(full_raw_text)
        self.vocab = sorted(list(set(self.tokens)))
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def tokenize(self, text):
        pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")
        return pattern.findall(text.lower())

    def generate_training_data(self, window_size):
        training_data = []

        for i in range(window_size, len(self.tokens) - window_size):
            target_word = self.tokens[i]

            context_words = [
                self.tokens[j]
                for j in range(i - window_size, i + window_size + 1)
                if j != i
            ]

            target_index = self.word_to_index[target_word]
            context_indices = [self.word_to_index[word] for word in context_words]

            training_data.append((context_indices, target_index))
        return training_data[2:-2]


SETTINGS = {
    "window_size": 2,                   # context window +/- in regard of center word
    "n": 50,                            # dimensions of word embeddings
    "epochs": 10,                       # number of training epochs
    "learning_rate": 0.01,              # learning rate
    "negative_sample_amount": 10,       # number of negative samples per positive sample
}

data_loader = DataLoader(["blindsight.txt"])
VOCAB = data_loader.vocab
TRAINING_DATA = data_loader.generate_training_data(SETTINGS["window_size"])


class Word2Vec:
    """
    CBOW implementation of Word2Vec.
    With Negative Sampling. (? not yet)
    """

    def __init__(self, vocab=VOCAB, settings=SETTINGS):
        self.window_size = settings["window_size"]
        self.dim = settings["n"]
        self.learning_rate = settings["learning_rate"]
        self.epochs = settings["epochs"]
        self.negative_sample_amount = settings.get("negative_sample_amount", 5)

        self.vocab = list(set(vocab))
        self.vocab_size = len(self.vocab)

        self.embedding_layer = np.random.uniform(-1/self.dim, 1/self.dim, (self.vocab_size, self.dim))
        self.context_layer = np.random.uniform(-1/self.dim, 1/self.dim, (self.vocab_size, self.dim))

    def forward(self, x):
        hidden = np.dot(self.embedding_layer.T, x)
        output = np.dot(self.context_layer.T, hidden)
        out_prob = self.softmax(output)
        return out_prob

    def backward(self, context_indices, hidden, error):
        d_context = np.outer(hidden, error)
        self.context_layer -= self.learning_rate * d_context
        hidden_error = np.dot(self.context_layer, error)
        for idx in context_indices:
            self.embedding_layer[idx] -= self.learning_rate * (hidden_error / len(context_indices))

    def forward_ns(self, context_indices, target_index, negative_indices):
        hidden = np.mean(self.embedding_layer[context_indices], axis=0)
        pos_score = np.dot(self.context_layer[target_index], hidden)
        neg_scores = np.dot(self.context_layer[negative_indices], hidden)

        return hidden, pos_score, neg_scores
    
    def backward_ns(self, context_indices, target_index, negative_indices, hidden, pos_score, neg_scores):
        e_pos = self.sigmoid(pos_score) - 1
        e_neg = self.sigmoid(neg_scores) - 0
        
        grad_pos = e_pos * hidden
        self.context_layer[target_index] -= self.learning_rate * grad_pos
        
        grad_neg = np.outer(e_neg, hidden)
        self.context_layer[negative_indices] -= self.learning_rate * grad_neg
        
        hidden_error = e_pos * self.context_layer[target_index] + np.dot(e_neg, self.context_layer[negative_indices])
        
        for idx in context_indices:
            self.embedding_layer[idx] -= self.learning_rate * (hidden_error / len(context_indices))

    def train(self, training_data):
        for epoch in range(self.epochs):
            loss = 0
            ITER = 0
            for context_indices, target_idx in training_data:
                ITER += 1
                hidden = np.mean(self.embedding_layer[context_indices], axis=0)
                u = np.dot(self.context_layer.T, hidden)
                y_pred = self.softmax(u)

                error = y_pred.copy()
                error[target_idx] -= 1

                self.backward(context_indices, hidden, error)

                loss -= np.log(y_pred[target_idx] + 1e-9)
                print(f"[NAIVE] Epoch {epoch+1}/{self.epochs} | Iteration {ITER}/{len(training_data)} | Loss: {loss:.4f}", end="\r")
            print("\r" + " " * 200 + "\r", end="")
            print("--------------------------------------------------------")
            print(f"[NAIVE] Epoch {epoch+1} complete. Loss: {loss:.4f}")
            print("--------------------------------------------------------")

    def train_ns(self, training_data):
        for epoch in range(self.epochs):
            loss = 0
            ITER = 0
            for context_indices, target_idx in training_data:
                ITER += 1
                negative_indices = np.random.choice(self.vocab_size, size=self.negative_sample_amount, replace=False)
                hidden, pos_score, neg_scores = self.forward_ns(context_indices, target_idx, negative_indices)
                self.backward_ns(context_indices, target_idx, negative_indices, hidden, pos_score, neg_scores)

                pos_loss = -np.log(self.sigmoid(pos_score) + 1e-9)
                neg_loss = -np.sum(np.log(1 - self.sigmoid(neg_scores) + 1e-9))
                loss += pos_loss + neg_loss

                print(f"[NEGATIVE SAMPLING] Epoch {epoch+1}/{self.epochs} | Iteration {ITER}/{len(training_data)} | Loss: {loss:.4f}\r", end="\r")

            print("\r" + " " * 200 + "\r", end="")
            print("--------------------------------------------------------")
            print(f"[NEGATIVE SAMPLING] Epoch {epoch+1} complete. Loss: {loss:.4f}", end="\n")
            print("--------------------------------------------------------")

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)
    
    def sigmoid(self, x):
        x = np.clip(x, -45, 45)
        return 1 / (1 + np.exp(-x))
    
    def get_embedding(self, word):
        idx = self.vocab.index(word)
        return self.embedding_layer[idx]
    
    def most_similar(self, word, top_n=5):
        if word not in self.vocab:
            raise ValueError(f"'{word}' not in vocabulary.")
        
        word_idx = self.vocab.index(word)
        word_embedding = self.embedding_layer[word_idx]
        
        similarities = np.dot(self.embedding_layer, word_embedding) / (np.linalg.norm(self.embedding_layer, axis=1) * np.linalg.norm(word_embedding) + 1e-9)
        
        most_similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        return [(self.vocab[idx], similarities[idx]) for idx in most_similar_indices]
    
    def distance(self, word1, word2):
        if word1 not in self.vocab or word2 not in self.vocab:
            raise ValueError(f"'{word1}' or '{word2}' not in vocabulary.")
        
        idx1 = self.vocab.index(word1)
        idx2 = self.vocab.index(word2)
        
        embedding1 = self.embedding_layer[idx1]
        embedding2 = self.embedding_layer[idx2]
        
        cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-9)
        return cosine_similarity


model = Word2Vec()
model.train_ns(TRAINING_DATA)
print(f"Most similar to 'blindsight': {model.most_similar('blindsight')}")
print(f"Most similar to 'fish': {model.most_similar('fish')}")