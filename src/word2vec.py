import json
import os

import numpy as np

from utils import Data, Settings


class Word2Vec:
    """
    CBOW implementation of Word2Vec.

    Includes:
    - Naive Softmax
    - Negative Sampling:

    Potential Optimizations:
    - Adam Optimizer: Handles updates of rare words better
    - Subsampling: Deals with frequent words (e.g., 'the', 'is')
    """

    def __init__(self, data: Data, settings: Settings):
        self.window_size = settings.window_size
        self.dim = settings.n
        self.learning_rate = settings.learning_rate
        self.epochs = settings.epochs
        self.negative_sample_amount = settings.negative_sample_amount

        self.vocab = data.vocab
        self.vocab_size = data.vocab_size
        self.word_to_index = data.word_to_index
        self.index_to_word = data.index_to_word

        self.embedding_layer = np.random.uniform(
            -1 / self.dim, 1 / self.dim, (self.vocab_size, self.dim)
        )
        self.context_layer = np.random.uniform(
            -1 / self.dim, 1 / self.dim, (self.vocab_size, self.dim)
        )

        self.filename = settings.save_file

    def _forward(self, context_indices):
        hidden = np.mean(self.embedding_layer[context_indices], axis=0)
        output = np.dot(self.context_layer, hidden)
        prob = self._softmax(output)
        return prob, hidden

    def _backprop(self, context_indices, hidden, error):
        grad_context = np.outer(error, hidden)
        self.context_layer -= self.learning_rate * grad_context

        hidden_error = np.dot(self.context_layer.T, error)
        grad_embedding = hidden_error / len(context_indices)

        for idx in context_indices:
            self.embedding_layer[idx] -= self.learning_rate * grad_embedding

    def train(self, training_data: list[tuple[list[int], int]]) -> None:
        print("--------------------------------------------------------")
        for epoch in range(self.epochs):
            LOSS = 0
            ITER = 0
            for context_indices, target_idx in training_data:
                ITER += 1
                pred, hidden = self._forward(context_indices)

                error = pred.copy()
                error[target_idx] -= 1
                # y_pred - y_true

                self._backprop(context_indices, hidden, error)

                # Cross entropy loss
                LOSS -= np.log(pred[target_idx] + 1e-9)

                print(
                    f"[NAIVE] Epoch {epoch+1}/{self.epochs} | Iteration {ITER}/{len(training_data)} | Loss: {LOSS:.4f}",
                    end="\r",
                )
                print("\033[K", end="")
            print(f"[NAIVE] Epoch {epoch+1} complete. Loss: {LOSS:.4f}")
            print("--------------------------------------------------------")

    def _forward_ns(self, context_indices, target_index, negative_indices):
        hidden = np.mean(self.embedding_layer[context_indices], axis=0)
        target_prob = np.dot(self.context_layer[target_index], hidden)
        negative_probs = np.dot(self.context_layer[negative_indices], hidden)

        return hidden, target_prob, negative_probs

    def _backprop_ns(
        self,
        context_indices,
        target_index,
        negative_indices,
        hidden,
        target_prob,
        negative_probs,
    ):
        # dLoss / dprob is sig(prob) - prob_true
        e_target = self._sigmoid(target_prob) - 1
        e_negatives = self._sigmoid(negative_probs) - 0

        grad_target = e_target * hidden
        self.context_layer[target_index] -= self.learning_rate * grad_target

        grad_negatives = np.outer(e_negatives, hidden)
        self.context_layer[negative_indices] -= self.learning_rate * grad_negatives

        hidden_error = e_target * self.context_layer[target_index] + np.dot(
            e_negatives, self.context_layer[negative_indices]
        )

        for idx in context_indices:
            self.embedding_layer[idx] -= self.learning_rate * (
                hidden_error / len(context_indices)
            )

    def train_ns(self, training_data: list[tuple[list[int], int]]) -> None:
        print("--------------------------------------------------------")
        for epoch in range(self.epochs):
            LOSS = 0
            ITER = 0
            for context_indices, target_idx in training_data:
                ITER += 1

                negative_indices = np.random.choice(
                    self.vocab_size, size=self.negative_sample_amount, replace=False
                )

                hidden, pos_score, neg_scores = self._forward_ns(
                    context_indices, target_idx, negative_indices
                )
                self._backprop_ns(
                    context_indices,
                    target_idx,
                    negative_indices,
                    hidden,
                    pos_score,
                    neg_scores,
                )

                # Binary Cross entropy
                target_loss = -np.log(self._sigmoid(pos_score) + 1e-9)
                negatives_loss = -np.sum(np.log(1 - self._sigmoid(neg_scores) + 1e-9))
                LOSS += target_loss + negatives_loss
                print(
                    f"[NEGATIVE SAMPLING] Epoch {epoch+1}/{self.epochs} | Iteration {ITER}/{len(training_data)} | Loss: {LOSS:.4f}",
                    end="\r",
                )
                print("\033[K", end="")
            print(f"[NEGATIVE SAMPLING] Epoch {epoch+1} complete. Loss: {LOSS:.4f}")
            print("--------------------------------------------------------")

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def _sigmoid(self, x):
        x = np.clip(x, -45, 45)
        return 1 / (1 + np.exp(-x))

    def get_embedding(self, word: str) -> np.ndarray:
        if word not in self.vocab:
            raise ValueError(f"'{word}' not in vocabulary.")

        idx = self.vocab.index(word)
        return self.embedding_layer[idx]

    def most_similar(self, word: str, top_n: int) -> list[tuple[str, float]]:
        if word not in self.vocab:
            raise ValueError(f"'{word}' not in vocabulary.")

        word_idx = self.vocab.index(word)
        word_embedding = self.embedding_layer[word_idx]

        diff = self.embedding_layer - word_embedding
        distances = np.linalg.norm(diff, axis=1)

        most_similar_indices = np.argsort(distances)[1 : top_n + 1]

        return [
            (self.vocab[idx], float(distances[idx])) for idx in most_similar_indices
        ]

    def distance(self, word1: str, word2: str) -> float:
        if word1 not in self.vocab or word2 not in self.vocab:
            raise ValueError(f"'{word1}' or '{word2}' not in vocabulary.")

        idx1 = self.vocab.index(word1)
        idx2 = self.vocab.index(word2)

        embedding1 = self.embedding_layer[idx1]
        embedding2 = self.embedding_layer[idx2]

        distance = float(np.linalg.norm(embedding1 - embedding2))
        return distance

    def save(self) -> None:
        np.save(f"{self.filename}_weights.npy", self.embedding_layer)
        metadata = {
            "vocab": self.vocab,
            "word_to_index": self.word_to_index,
            "dim": self.dim,
        }
        with open(f"{self.filename}_metadata.json", "w") as f:
            json.dump(metadata, f)
        print(f"Model saved to: \n  {self.filename}_ metadata.json \n  {self.filename}_ weights.npy")

    def load(self) -> None:
        if not os.path.exists(f"{self.filename}_metadata.json"):
            print(f"Error: No model found at {self.filename}")
            return

        if not os.path.exists(f"{self.filename}_weights.npy"):
            print(f"Error: No model found at {self.filename}")
            return

        with open(f"{self.filename}_metadata.json", "r") as f:
            metadata = json.load(f)

        self.vocab = metadata["vocab"]
        self.word_to_index = metadata["word_to_index"]
        self.dim = metadata["dim"]
        self.vocab_size = len(self.vocab)

        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}

        self.embedding_layer = np.load(f"{self.filename}_weights.npy")
