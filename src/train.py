from utils import Data, DataLoader, Settings
from word2vec import Word2Vec

SETTINGS = Settings(
    window_size=2,
    n=50,
    epochs=20,
    learning_rate=0.01,
    negative_sample_amount=5,
    save_file="embd",
)
DATA = Data(DataLoader(["../data/alice.txt"]))
TRAINING_DATA = DATA.generate_training_data(SETTINGS.window_size)

model = Word2Vec(DATA, SETTINGS)
model.train_ns(TRAINING_DATA)
word = "Alice"
print(f"Most similar to '{word.lower()}': {model.most_similar(word.lower(), top_n= 5)}")


save = input("Save embeddings? [Y/N] ")
if save and save.lower() == "y":
    model.save()
else:
    print("Embeddings not saved.")
