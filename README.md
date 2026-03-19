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