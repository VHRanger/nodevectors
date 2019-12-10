[![Build Status](https://travis-ci.com/VHRanger/graph2vec.svg?branch=master)](https://travis-ci.com/VHRanger/graph2vec)

## Quick Example:
```python
    import networkx as nx
    from graph2vec import Node2Vec

    # Test Graph
    G = nx.generators.classic.wheel_graph(100)
 
    # Fit embedding model to graph
    g2v = Node2Vec()
    g2v.fit(G) # way faster than other node2vec implementations
 
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

Currently through downloading this project and `python setup.py install`. I'll get it on pip ASAP.

#### Test Installation (from project's root folder)

    python -m unittest discover tests
