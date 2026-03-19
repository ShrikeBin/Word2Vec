# Word2Vec

```bash
Task #1

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). 
The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure 
(forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

The submitted solution should be fully understood by the applicant: 
during follow-up we will ask questions about the ideas behind word2vec, 
the code, gradient derivations, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.
```

# Structure
```
word2vec/
├── data/
│   └── alice.txt       # Example data
├── src/
│   ├── model.py        # Word2Vec class
│   ├── utils.py        # Util classes for data loading etc.
│   └── train.py        # Trainig script
└── README.md               
```

# Requirements:

- numpy
- python 3.12+

# Example:

```bash
...
--------------------------------------------------------
[NEGATIVE SAMPLING] Epoch 10 complete. Loss: 55583.9354
--------------------------------------------------------
[NEGATIVE SAMPLING] Epoch 11 complete. Loss: 55033.5484
--------------------------------------------------------
[NEGATIVE SAMPLING] Epoch 12 complete. Loss: 54421.1110
--------------------------------------------------------
Most similar to 'alice': 
[('you', 0.41745737422326085), 
 ('it', 0.43266910769314604), 
 ('and', 0.5745998686287821), 
 ('said', 0.5821859805125168), 
 ('she', 0.7022842829413142)]
Save embeddings? [Y/N] Y
Model saved to: 
  alice_ metadata.json 
  alice_ weights.npy
```