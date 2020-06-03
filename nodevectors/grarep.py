import numpy as np
from sklearn.decomposition import TruncatedSVD

import csrgraph as cg
from nodevectors.embedders import BaseNodeEmbedder


def grarep_sum_merger(x):
    """default pooling method for GraRep (summing along axes)"""
    return np.sum(x, axis=0)

class GraRep(BaseNodeEmbedder):
    def __init__(self, 
        n_components=32,
        order=2,
        embedder=TruncatedSVD(
            n_iter=10,
            random_state=42),
        merger=grarep_sum_merger,
        verbose=True):
        """
        Arbitrarily high order global embedding

        Embeddings of powers of the PMI matrix of the graph adj matrix.

        NOTE: Unlike GGVec and GLoVe, the base method returns a LIST OF EMBEDDINGS
            (one per order). 
        The `merger` parameter is the pooling of these embeddings. 
        Default pooling is to sum : 
            `lambda x : np.sum(x, axis=0)`
        You can also take only the highest order embedding:
            `lambda x : x[-1]`
        Etc.

        Original paper: https://dl.acm.org/citation.cfm?id=2806512

        Parameters : 
        ----------------
        n_components (int): 
            Number of individual embedding dimensions.
        order (int): 
            Number of PMI matrix powers.
            The performance degrades close to quadratically as a factor of this parameter.
            Generally should be kept under 5.
        embedder : (instance of sklearn API compatible model)
            Should implement the `fit_transform` method: 
                https://scikit-learn.org/stable/glossary.html#term-fit-transform
            The model should also have `n_components` as a parameter
            for number of resulting embedding dimensions. See:
                https://scikit-learn.org/stable/modules/manifold.html#manifold
            If not compatible, set resulting dimensions in the model instance directly
        merger : function[list[array]] -> array
            GraRep returns one embedding matrix per order.
            This function reduces it to a single matrix.
        """
        self.n_components = n_components
        self.order = order
        self.embedder = embedder
        self.merger = merger
        self.verbose = verbose

    def fit(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph)
        vectors = G.grarep(n_components=self.n_components, 
                           order=self.order, embedder=self.embedder,   
                           verbose=self.verbose)
        vectors = self.merger(vectors)
        self.model = dict(zip(G.nodes(), vectors))

    def fit_transform(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph)
        vectors = G.grarep(n_components=self.n_components, 
                           order=self.order, embedder=self.embedder,   
                           verbose=self.verbose)
        vectors = self.merger(vectors)
        self.model = dict(zip(G.nodes(), vectors))
        return vectors
