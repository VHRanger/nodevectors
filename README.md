## Quick Example:

    import networkx as nx
    from graph2vec import Node2Vec

    # Test Graph
    G = nx.generators.classic.wheel_graph(100)
 
    # Fit embedding model to graph
    g2v = Node2Vec()
    g2v.fit(G)
 
    # query embeddings for node 42
    g2v.predict(42)

    # Save model to gensim.KeyedVector format
    g2v.save("wheel_model.bin")
    
    # Load directly in gensim
    from gensim.models import KeyedVectors
    kv = KeyedVectors.load_word2vec_format("wheel_model.bin")
    kv[str(42)] # need to convert node ID to str explicitly to read in gensim

## Installing

### Dependencies

- Numba
- Gensim
- Pandas
- Scipy
- Numpy


#### Test Installation (from project's root folder)

    python -m unittest discover tests
