import networkx as nx
import numba
from numba import jit
import numpy as np
import os
import pandas as pd
from scipy import sparse
import time
import warnings

### TODO: could drop gensim dependency by making coocurence matrix on the fly
###       instead of random walks and use GLoVe on it.
###       but then why not just use GLoVe on Transition matrix?
# Gensim triggers automatic useless warnings for windows users...
warnings.simplefilter("ignore", category=UserWarning)
import gensim
warnings.resetwarnings()


# TODO: Organize Graph method here
# Layout nodes by their 1d embedding's position


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _csr_random_walk(Tdata, Tindptr, Tindices,
                    sampling_nodes,
                    walklen):
    """
    Create random walks from the transition matrix of a graph 
        in CSR sparse format

    NOTE: scales linearly with threads but hyperthreads don't seem to 
            accelerate this linearly
    
    Parameters
    ----------
    Tdata : 1d np.array
        CSR data vector from a sparse matrix. Can be accessed by M.data
    Tindptr : 1d np.array
        CSR index pointer vector from a sparse matrix. 
        Can be accessed by M.indptr
    Tindices : 1d np.array
        CSR column vector from a sparse matrix. 
        Can be accessed by M.indices
    sampling_nodes : 1d np.array of int
        List of node IDs to start random walks from.
        Is generally equal to np.arrange(n_nodes) repeated for each epoch
    walklen : int
        length of the random walks

    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a random walk, 
        and each entry is the ID of the node
    """
    n_walks = len(sampling_nodes)
    res = np.empty((n_walks, walklen), dtype=np.int64)
    for i in numba.prange(n_walks):
        # Current node (each element is one walk's state)
        state = sampling_nodes[i]
        for k in range(walklen-1):
            # Write state
            res[i, k] = state
            # Find row in csr indptr
            start = Tindptr[state]
            end = Tindptr[state+1]
            # transition probabilities
            p = Tdata[start:end]
            # cumulative distribution of transition probabilities
            cdf = np.cumsum(p)
            # Random draw in [0, 1] for each row
            # Choice is where random draw falls in cumulative distribution
            draw = np.random.rand()
            # Find where draw is in cdf
            # Then use its index to update state
            next_idx = np.searchsorted(cdf, draw)
            # Winner points to the column index of the next node
            state = Tindices[start + next_idx]
        # Write final states
        res[i, -1] = state
    return res


# TODO: This throws heap corruption errors when made parallel
#       doesn't seem to be illegal reads anywhere though...
@jit(nopython=True, nogil=True, fastmath=True)
def _csr_node2vec_walks(Tdata, Tindptr, Tindices,
                        sampling_nodes,
                        walklen,
                        return_weight,
                        neighbor_weight):
    """
    Create biased random walks from the transition matrix of a graph 
        in CSR sparse format. Bias method comes from Node2Vec paper.
    
    Parameters
    ----------
    Tdata : 1d np.array
        CSR data vector from a sparse matrix. Can be accessed by M.data
    Tindptr : 1d np.array
        CSR index pointer vector from a sparse matrix. 
        Can be accessed by M.indptr
    Tindices : 1d np.array
        CSR column vector from a sparse matrix. 
        Can be accessed by M.indices
    sampling_nodes : 1d np.array of int
        List of node IDs to start random walks from.
        Is generally equal to np.arrange(n_nodes) repeated for each epoch
    walklen : int
        length of the random walks
    return_weight : float in (0, inf]
        Weight on the probability of returning to node coming from
        Having this higher tends the walks to be 
        more like a Breadth-First Search.
        Having this very high  (> 2) makes search very local.
        Equal to the inverse of p in the Node2Vec paper.
    neighbor_weight : float in (0, inf]
        Weight on the probability of visitng a neighbor node
        to the one we're coming from in the random walk
        Having this higher tends the walks to be 
        more like a Depth-First Search.
        Having this very high makes search more outward.
        Having this very low makes search very local.
        Equal to the inverse of q in the Node2Vec paper.
    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a biased random walk, 
        and each entry is the ID of the node
    """
    n_walks = len(sampling_nodes)
    res = np.empty((n_walks, walklen), dtype=np.int64)
    for i in range(n_walks):
        # Current node (each element is one walk's state)
        state = sampling_nodes[i]
        res[i, 0] = state
        # Do one normal step first
        # comments for these are in _csr_random_walk
        start = Tindptr[state]
        end = Tindptr[state+1]
        p = Tdata[start:end]
        cdf = np.cumsum(p)
        draw = np.random.rand()
        next_idx = np.searchsorted(cdf, draw)
        state = Tindices[start + next_idx]
        for k in range(1, walklen-1):
            # Write state
            res[i, k] = state
            # Find rows in csr indptr
            prev = res[i, k-1]
            start = Tindptr[state]
            end = Tindptr[state+1]
            start_prev = Tindptr[prev]
            end_prev = Tindptr[prev+1]
            # Find overlaps and fix weights
            this_edges =  Tindices[start:end]
            prev_edges =  Tindices[start_prev:end_prev]
            p = np.copy(Tdata[start:end])
            ret_idx = np.where(this_edges == prev)
            p[ret_idx] = np.multiply(p[ret_idx], return_weight)
            for pe in prev_edges:
                n_idx = np.where(this_edges == pe)[0]
                p[n_idx] = np.multiply(p[n_idx], neighbor_weight)
            # Get next state
            cdf = np.cumsum(np.divide(p, np.sum(p)))
            draw = np.random.rand()
            next_idx = np.searchsorted(cdf, draw)
            state = this_edges[next_idx]
        # Write final states
        res[i, k] = state
    return res


def make_walks(T,
               walklen=10,
               epochs=3,
               return_weight=1.,
               neighbor_weight=1.,
               threads=0):
    """
    Create random walks from the transition matrix of a graph 
        in CSR sparse format

    NOTE: scales linearly with threads but hyperthreads don't seem to 
            accelerate this linearly

    Parameters
    ----------
    T : scipy.sparse.csr matrix
        Graph transition matrix in CSR sparse format
    walklen : int
        length of the random walks
    epochs : int
        number of times to start a walk from each nodes
    return_weight : float in (0, inf]
        Weight on the probability of returning to node coming from
        Having this higher tends the walks to be 
        more like a Breadth-First Search.
        Having this very high  (> 2) makes search very local.
        Equal to the inverse of p in the Node2Vec paper.
    neighbor_weight : float in (0, inf]
        Weight on the probability of visitng a neighbor node
        to the one we're coming from in the random walk
        Having this higher tends the walks to be 
        more like a Depth-First Search.
        Having this very high makes search more outward.
        Having this very low makes search very local.
        Equal to the inverse of q in the Node2Vec paper.
    threads : int
        number of threads to use.  0 is full use

    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a random walk, 
        and each entry is the ID of the node
    """
    n_rows = T.shape[0]
    sampling_nodes = np.arange(n_rows)
    sampling_nodes = np.tile(sampling_nodes, epochs)
    if type(threads) is not int:
        raise ValueError("Threads argument must be an int!")
    if threads == 0:
        threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    threads = str(threads)
    try:
        prev_numba_value = os.environ['NUMBA_NUM_THREADS']
    except KeyError:
        prev_numba_value = threads
    # If we change the number of threads, recompile
    if threads != prev_numba_value:
        os.environ['NUMBA_NUM_THREADS'] = threads
        _csr_node2vec_walks.recompile()
        _csr_random_walk.recompile()
    if return_weight <= 0 or neighbor_weight <= 0:
        raise ValueError("Return and neighbor weights must be > 0")
    if (return_weight > 1. or  return_weight < 1. 
            or neighbor_weight < 1. or neighbor_weight > 1.):
        walks = _csr_node2vec_walks(T.data, T.indptr, T.indices, 
                                    sampling_nodes=sampling_nodes, 
                                    walklen=walklen, 
                                    return_weight=return_weight, 
                                    neighbor_weight=neighbor_weight)
    # much faster implementation for regular walks
    else:
        walks = _csr_random_walk(T.data, T.indptr, T.indices, 
                                 sampling_nodes, walklen)
    # set back to default
    os.environ['NUMBA_NUM_THREADS'] = prev_numba_value
    return walks



def _sparse_normalize_rows(mat):
    """
    Normalize a sparse CSR matrix row-wise (each row sums to 1)

    If a row is all 0's, it remains all 0's
    
    Parameters
    ----------
    mat : scipy.sparse.csr matrix
        Matrix in CSR sparse format

    Returns
    -------
    out : scipy.sparse.csr matrix
        Normalized matrix in CSR sparse format
    """
    n_nodes = mat.shape[0]
    # Normalize Adjacency matrix to transition matrix
    # Diagonal of the degree matrix is the sum of nonzero elements
    degrees_div = np.array(np.sum(mat, axis=1)).flatten()
    # This is equivalent to inverting the diag mat
    # weights are 1 / degree
    degrees = np.divide(
        1,
        degrees_div,
        out=np.zeros_like(degrees_div, dtype=float),
        where=(degrees_div != 0)
    )
    # construct sparse diag mat 
    # to broadcast weights to adj mat by dot product
    D = sparse.dia_matrix((n_nodes,n_nodes), dtype=np.float64)
    D.setdiag(degrees)   
    # premultiplying by diag mat is row-wise mul
    return sparse.csr_matrix(D.dot(mat))



class Node2Vec():
    """
    Embeds NetworkX into a continuous representation of the nodes.

    The resulting embedding can be queried just like word embeddings.

    Note: the graph's node names need to be int or str.
    """
    def __init__(
        self, walklen=10, epochs=20, return_weight=1., 
        neighbor_weight=1., threads=0, keep_walks=True,
        w2vparams={"window":10, "size":32, "negative":20, "iter":10,
                   "batch_words":128}):
        """
        Parameters
        ----------
        walklen : int
            length of the random walks
        epochs : int
            number of times to start a walk from each nodes
        return_weight : float in (0, inf]
            Weight on the probability of returning to node coming from
            Having this higher tends the walks to be 
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        neighbor_weight : float in (0, inf]
            Weight on the probability of visitng a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be 
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        threads : int
            number of threads to use. 0 is full use
        w2vparams : dict
            dictionary of parameters to pass to gensim's word2vec
            of relevance is "size" (length of resulting embedding vector)
        """
        if type(threads) is not int:
            raise ValueError("Threads argument must be an int!")
        if walklen < 1 or epochs < 1:
            raise ValueError("Walklen and epochs arguments must be > 1")
        if return_weight < 0 or neighbor_weight < 0:
            raise ValueError("return_weight and neighbor_weight must be >= 0")
        self.walklen = walklen
        self.epochs = epochs
        self.return_weight = return_weight
        self.neighbor_weight = neighbor_weight
        self.keep_walks = keep_walks
        self.w2vparams = w2vparams
        if threads == 0:
            threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
        self.threads = threads
        w2vparams['workers'] = threads

    def fit(self, nxGraph: nx.Graph, verbose=1):
        """
        NOTE: Currently only support str as node name for graph
        Parameters
        ----------
        nxGraph : NetworkX.Graph
            NetworkX graph to embed
        verbose : bool
            Whether to print output while working
        """
        # Because networkx graphs are actually iterables of their nodes
        #   we do list(G) to avoid networkx 1.X vs 2.X errors
        node_names = list(nxGraph)
        if type(node_names[0]) not in [int, str, np.int32, np.int64]:
            raise ValueError("Graph node names must be int or str!")
        # Adjacency matrix
        A = nx.adj_matrix(nxGraph)
        n_nodes = A.shape[0]
        T = _sparse_normalize_rows(A)
        walks_t = time.time()
        if verbose:
            print("Making walks...", end=" ")
                # If node2vec graph weights not identity, apply them
        self.walks = make_walks(T, walklen=self.walklen, epochs=self.epochs,
                            return_weight=self.return_weight,
                            neighbor_weight=self.neighbor_weight,
                            threads=self.threads)
        if verbose:
            print(f"Done, T={time.time() - walks_t:.2f}")
            print("Mapping Walk Names...", end=" ")
        map_t = time.time()
        self.walks = pd.DataFrame(self.walks)
        # Map nodeId -> node name
        node_dict = dict(zip(np.arange(n_nodes), node_names))
        for col in self.walks.columns:
            self.walks[col] = self.walks[col].map(node_dict).astype(str)
        self.walks = [list(x) for x in self.walks.itertuples(False, None)]
        if verbose:
            print(f"Done, T={time.time() - map_t:.2f}")
            print("Training W2V...", end=" ")
            if gensim.models.word2vec.FAST_VERSION < 1:
                print("WARNING: gensim word2vec version is unoptimized"
                    "Try version 3.6 if on windows, versions 3.7 "
                    "and 3.8 have had issues")
        w2v_t = time.time()
        # Train gensim word2vec model on random walks
        self.model = gensim.models.Word2Vec(
            sentences=self.walks,
            **self.w2vparams)
        if not self.keep_walks:
            del self.walks
        if verbose:
            print(f"Done, T={time.time() - w2v_t:.2f}")
    
    def predict(self, node_name):
        """
        Return vector associated with node
        """
        # current hack to work around word2vec problem
        # ints need to be str -_-
        if type(node_name) is not str:
            node_name = str(node_name)
        return self.model.wv.__getitem__(node_name)

    def save(self, out_file):
        """
        Save as embeddings in gensim.models.KeyedVectors format
        """
        self.model.wv.save_word2vec_format(out_file)

    def load(self, out_file):
        """
        Load embeddings from gensim.models.KeyedVectors format
        """
        self.model = gensim.wv.load_word2vec_format(out_file)
