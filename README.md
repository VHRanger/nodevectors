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

    # Save and load whole node2vec model
    # Uses a smart pickling method to avoid serialization errors
    g2v.save('node2vec.pckl')
    g2v = Node2vec.load('node2vec.pckl')
    
    # Save model to gensim.KeyedVector format
    g2v.save_vectors("wheel_model.bin")
    
    # load in gensim
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format("wheel_model.bin")
    model[str(43)] # need to make nodeID a str for gensim
    
```
## Installing

`pip install nodevectors`

### Usage

The public methods are all exposed in the quick example. The documentation is included in the docstrings of the methods, so for instance typing `g2v.fit?` in a Jupyter Notebook will expose the documentation directly.

## Why is it so fast?

We leverage [CSRGraphs](https://github.com/VHRanger/CSRGraph) to do the random walks. This uses CSR graph representations and a lot of Numba usage.
