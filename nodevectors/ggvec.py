import csrgraph as cg
from nodevectors.embedders import BaseNodeEmbedder

class GGVec(BaseNodeEmbedder):
    def __init__(self, 
        n_components=32,
        order=1,
        learning_rate=0.1, max_loss=10.,
        tol="auto", tol_samples=30,
        exponent=0.33,
        threads=0,
        negative_ratio=0.15,
        max_epoch=350, 
        verbose=False):
        """
        GGVec: Fast global first (and higher) order local embeddings.

        This algorithm directly minimizes related nodes' distances.
        It uses a relaxation pass (negative sample) + contraction pass (loss minimization)
        To find stable embeddings based on the minimal dot product of edge weights.

        Parameters:
        -------------
        n_components (int): 
            Number of individual embedding dimensions.
        order : int >= 1
            Meta-level of the embeddings. Improves link prediction performance.
            Setting this higher than 1 ~quadratically slows down algorithm
                Order = 1 directly optimizes the graph.
                Order = 2 optimizes graph plus neighbours of neighbours
                Order = 3 optimizes up to 3rd order edges
                (and so on)
            Higher order edges are automatically weighed using GraRep-style graph formation
            Eg. the higher-order graph is from stable high-order random walk distribution.
        negative_ratio : float in [0, 1]
            Negative sampling ratio.
            Setting this higher will do more negative sampling.
            This is slower, but can lead to higher quality embeddings.
        exponent : float
            Weighing exponent in loss function. 
            Having this lower reduces effect of large edge weights.
        tol : float in [0, 1] or "auto"
            Optimization early stopping criterion.
            Stops average loss < tol for tol_samples epochs.
            "auto" sets tol as a function of learning_rate
        tol_samples : int
            Optimization early stopping criterion.
            This is the number of epochs to sample for loss stability.
            Once loss is stable over this number of epochs we stop early.
        negative_decay : float in [0, 1]
            Decay on negative ratio.
            If >0 then negative ratio will decay by (1-negative_decay) ** epoch
            You should usually leave this to 0.
        max_epoch : int
            Stopping criterion.
        max_count : int
            Ceiling value on edge weights for numerical stability
        learning_rate : float in [0, 1]
            Optimization learning rate.
        max_loss : float
            Loss value ceiling for numerical stability.
        """
        self.n_components = n_components
        self.tol = tol
        self.order=order
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.exponent = exponent
        self.max_loss = max_loss
        self.tol_samples = tol_samples
        self.threads = threads
        self.negative_ratio = negative_ratio
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
        vectors = G.ggvec(
            n_components=self.n_components, 
            order=self.order,
            exponent=self.exponent,
            tol=self.tol, max_epoch=self.max_epoch,
            learning_rate=self.learning_rate, 
            tol_samples=self.tol_samples,
            max_loss=self.max_loss,
            negative_ratio=self.negative_ratio,
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
        vectors = G.ggvec(
            n_components=self.n_components, 
            order=self.order,
            exponent=self.exponent,
            tol=self.tol, max_epoch=self.max_epoch,
            learning_rate=self.learning_rate, 
            tol_samples=self.tol_samples,
            max_loss=self.max_loss,
            negative_ratio=self.negative_ratio,
            verbose=self.verbose)
        self.model = dict(zip(G.nodes(), vectors))
        return vectors
