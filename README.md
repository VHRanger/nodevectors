[![Build Status](https://travis-ci.com/VHRanger/nodevectors.svg?branch=master)](https://travis-ci.com/VHRanger/nodevectors)

## Quick Example:
```python
    import networkx as nx
    from nodevectors import Node2Vec

    # Test Graph
    G = nx.generators.classic.wheel_graph(100)
 
    # Fit embedding model to graph
    g2v = Node2Vec()
    # way faster than other node2vec implementations
    # Graph edge weights are handled automatically
    g2v.fit(G)
 
    # query embeddings for node 42
    g2v.predict(42)

    # Save model to gensim.KeyedVector format
    g2v.save("wheel_model.bin")
    
    # load in gensim
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format("wheel_model.bin")
    model[str(43)] # need to make nodeID a str for gensim
    
```
## Installing

`pip install nodevectors`

### Usage

The public methods are all exposed in the quick example. The documentation is included in the docstrings of the methods, so for instance typing `g2v.fit?` in a Jupyter Notebook will expose the documentation directly.

## How does it work?

We transform the graph into a CSR sparse matrix, and generate the random walks directly on the CSR matrix raw data with optimized Numba JIT'ed code. After that, a Word2Vec model is trained on the random walks, as if the walks were the Word2Vec sentences.
