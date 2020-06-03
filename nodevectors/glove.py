import csrgraph as cg
from nodevectors.embedders import BaseNodeEmbedder

class Glove(BaseNodeEmbedder):
    def __init__(
        self, n_components=32,
        tol=0.0001, max_epoch=1000, 
        max_count=50, learning_rate=0.1, 
        max_loss=10., exponent=0.5,
        threads=0, verbose=True):
        """
        Global first order embedding for positive, count-valued sparse matrices.

        This algorithm is normally used in NLP on word co-occurence matrices.
        The algorithm fails if any value in the sparse matrix < 1.
        It is also a poor choice for matrices with homogeneous edge weights.

        Parameters:
        -------------
        n_components (int): 
            Number of individual embedding dimensions.
        tol : float in [0, 1]
            Optimization early stopping criterion.
            Stops when largest gradient is < tol
        max_epoch : int
            Stopping criterion.
        max_count : int
            Ceiling value on edge weights for numerical stability
        exponent : float
            Weighing exponent in loss function. 
            Having this lower reduces effect of large edge weights.
        learning_rate : float in [0, 1]
            Optimization learning rate.
        max_loss : float
            Loss value ceiling for numerical stability.

        References:
        -------------
        Paper: https://nlp.stanford.edu/pubs/glove.pdf
        Original implementation: https://github.com/stanfordnlp/GloVe/blob/master/src/glove.c
        """
        self.n_components = n_components
        self.tol = tol
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.max_loss = max_loss
        self.max_count = max_count
        self.threads = threads
        self.exponent = exponent
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
        G = cg.csrgraph(graph, threads=self.threads)
        vectors = G.glove(n_components=self.n_components, 
                              tol=self.tol, max_epoch=self.max_epoch,
                              learning_rate=self.learning_rate,
                              max_loss=self.max_loss,
                              max_count=self.max_count,
                              exponent=self.exponent,
                              verbose=self.verbose)
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
        G = cg.csrgraph(graph, threads=self.threads)
        vectors = G.glove(n_components=self.n_components, 
                              tol=self.tol, max_epoch=self.max_epoch,
                              learning_rate=self.learning_rate,
                              max_loss=self.max_loss,
                              max_count=self.max_count,
                              exponent=self.exponent,
                              verbose=self.verbose)
        self.model = dict(zip(G.nodes(), vectors))
        return vectors
