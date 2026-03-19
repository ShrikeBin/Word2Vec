import re
from dataclasses import dataclass


class DataLoader:
    def __init__(self, filepaths: list[str]):
        full_raw_text = ""
        for filepath in filepaths:
            with open(filepath, "r") as file:
                full_raw_text += file.read()

        self.tokens = self.tokenize(full_raw_text)

    def tokenize(self, text):
        pattern = re.compile(r"\w+(?:'\w+)?")
        return pattern.findall(text.lower())


class Data:
    def __init__(self, loader: DataLoader):
        self.tokens = loader.tokens
        self.vocab = sorted(list(set(self.tokens)))
        self.vocab_size = len(self.vocab)
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def generate_training_data(self, window_size: int) -> list[tuple[list[int], int]]:
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


@dataclass
class Settings:
    window_size: int            # context window +/- in regard of center word
    n: int                      # dimensions of word embeddings
    epochs: int                 # number of training epochs
    learning_rate: float        # learning rate
    negative_sample_amount: int # number of negative samples per positive sample
    save_file: str              # where to save embeddings
