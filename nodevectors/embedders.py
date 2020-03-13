import gc
import joblib
import json
import networkx as nx
import numba
from numba import jit
import numpy as np
import os
import pandas as pd
from scipy import sparse
import shutil
from sklearn.base import BaseEstimator
import sklearn
import sys
import tempfile
import time
import warnings

# Gensim triggers automatic useless warnings for windows users...
warnings.simplefilter("ignore", category=UserWarning)
import gensim
warnings.simplefilter("default", category=UserWarning)

import csrgraph
from csrgraph import CSRGraph

class BaseNodeEmbedder(BaseEstimator):
    """
    Base Class for node vector embedders
    Includes saving, loading, and predict_new methods
    """
    f_model = "model.pckl"
    f_mdata = "metadata.json"

    def save(self, filename: str):
        """
        Saves model to a custom file format
        
        filename : str
            Name of file to save. Don't include filename extensions
            Extensions are added automatically
        
        File format is a zipfile with joblib dump (pickle-like) + dependency metata
        Metadata is checked on load.
        
        Includes validation and metadata to avoid Pickle deserialization gotchas
        See here Alex Gaynor PyCon 2014 talk "Pickles are for Delis"
            for more info on why we introduce this additional check
        """
        if '.zip' in filename:
            raise UserWarning("The file extension '.zip' is automatically added"
                + " to saved models. The name will have redundant extensions")
        sysverinfo = sys.version_info
        meta_data = {
            "python_": f'{sysverinfo[0]}.{sysverinfo[1]}',
            "skl_": sklearn.__version__[:-2],
            "pd_": pd.__version__[:-2],
            "csrg_": csrgraph.__version__[:-2]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            joblib.dump(self, os.path.join(temp_dir, self.f_model), compress=True)
            with open(os.path.join(temp_dir, self.f_mdata), 'w') as f:
                json.dump(meta_data, f)
            filename = shutil.make_archive(filename, 'zip', temp_dir)

    @staticmethod
    def load(filename: str):
        """
        Load model from NodeEmbedding model zip file.
        
        filename : str
            full filename of file to load (including extensions)
            The file should be the result of a `save()` call
            
        Loading checks for metadata and raises warnings if pkg versions
        are different than they were when saving the model.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(filename, temp_dir, 'zip')
            model = joblib.load(os.path.join(temp_dir, BaseNodeEmbedder.f_model))
            with open(os.path.join(temp_dir, BaseNodeEmbedder.f_mdata)) as f:
                meta_data = json.load(f)
            # Validate the metadata
            sysverinfo = sys.version_info
            pyver = "{0}.{1}".format(sysverinfo[0], sysverinfo[1])
            if meta_data["python_"] != pyver:
                raise UserWarning(
                    "Invalid python version; {0}, required: {1}".format(
                        pyver, meta_data["python_"]))
            sklver = sklearn.__version__[:-2]
            if meta_data["skl_"] != sklver:
                raise UserWarning(
                    "Invalid sklearn version; {0}, required: {1}".format(
                        sklver, meta_data["skl_"]))
            pdver = pd.__version__[:-2]
            if meta_data["pd_"] != pdver:
                raise UserWarning(
                    "Invalid pandas version; {0}, required: {1}".format(
                        pdver, meta_data["pd_"]))
            csrv = csrgraph.__version__[:-2]
            if meta_data["csrg_"] != csrv:
                raise UserWarning(
                    "Invalid CSRGraph version; {0}, required: {1}".format(
                        csrv, meta_data["csrg_"]))
        return model

    def predict_new(self, neighbor_list, fn=np.mean):
        """
        Predicts a new node from its neighbors
        Generally done through the average of the embeddings
            of its neighboring nodes.
            
        neighbor_list : iterable[node_id or node name]
            list of nodes with edges to the new node to be predicted
        
        fn : function array -> scalar
            function to reduce the embeddings. Normally np.mean or np.sum
            This takes the embeddings of the neighbors and reduces them 
               to the final estimate for the unseen node
        """
        # TODO: test
        pass

class Node2Vec(BaseNodeEmbedder):
    """
    Embeds NetworkX into a continuous representation of the nodes.
    The resulting embedding can be queried just like word embeddings.
    Note: the graph's node names need to be int or str.
    """
    def __init__(
        self, 
        walklen=10, 
        epochs=20,
        return_weight=1.,
        neighbor_weight=1.,
        n_components=32,
        threads=0, 
        keep_walks=False,
        w2vparams={"window":10, "negative":5, "iter":10,
                   "batch_words":128}):
        """
        Parameters
        ----------
        walklen : int
            length of the random walks
        epochs : int
            number of times to start a walk from each nodes
        threads : int
            number of threads to use. 0 is full use
        n_components : int
            number of resulting dimensions for the embedding
            This should be set here rather than in the w2vparams arguments
        return_weight : float in (0, inf]
            Weight on the probability of returning to node coming from
            Having this higher tends the walks to be 
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight : float in (0, inf]
            Weight on the probability of visitng a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be 
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        keep_walks : bool
            Whether to save the random walks in the model object after training
        w2vparams : dict
            dictionary of parameters to pass to gensim's word2vec
            Don't set the embedding dimensions through arguments here.
        """
        if type(threads) is not int:
            raise ValueError("Threads argument must be an int!")
        if walklen < 1 or epochs < 1:
            raise ValueError("Walklen and epochs arguments must be > 1")
        self.n_components_ = n_components
        self.walklen = walklen
        self.epochs = epochs
        self.keep_walks = keep_walks
        if 'size' in w2vparams.keys():
            raise AttributeError("Embedding dimensions should not be set "
                + "through w2v parameters, but through n_components")
        self.w2vparams = w2vparams
        self.return_weight = return_weight
        self.neighbor_weight = neighbor_weight
        if threads == 0:
            threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
        self.threads = threads
        w2vparams['workers'] = threads

    def fit(self, nxGraph, verbose=1):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        verbose : bool
            Whether to print output while working
        """
        # Because networkx graphs are actually iterables of their nodes
        #   we do list(G) to avoid networkx 1.X vs 2.X errors
        node_names = list(nxGraph)
        G = CSRGraph(nxGraph, threads=self.threads)
        if type(node_names[0]) not in [int, str, np.int32, np.int64]:
            raise ValueError("Graph node names must be int or str!")
        # Adjacency matrix
        walks_t = time.time()
        if verbose:
            print("Making walks...", end=" ")
        self.walks = G.random_walks(walklen=self.walklen, 
                                    epochs=self.epochs,
                                    return_weight=self.return_weight,
                                    neighbor_weight=self.neighbor_weight)
        if verbose:
            print(f"Done, T={time.time() - walks_t:.2f}")
            print("Mapping Walk Names...", end=" ")
        map_t = time.time()
        self.walks = pd.DataFrame(self.walks)
        # Map nodeId -> node name
        node_dict = dict(zip(np.arange(len(node_names)), node_names))
        for col in self.walks.columns:
            self.walks[col] = self.walks[col].map(node_dict).astype(str)
        # Somehow gensim only trains on this list iterator
        # it silently mistrains on array input
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
            size=self.n_components_,
            **self.w2vparams)
        if not self.keep_walks:
            del self.walks
        if verbose:
            print(f"Done, T={time.time() - w2v_t:.2f}")
    
    def predict(self, node_name):
        """
        Return vector associated with node
        node_name : str or int
            either the node ID or node name depending on graph format
        """
        # current hack to work around word2vec problem
        # ints need to be str -_-
        if type(node_name) is not str:
            node_name = str(node_name)
        return self.model.wv.__getitem__(node_name)

    def save_vectors(self, out_file):
        """
        Save as embeddings in gensim.models.KeyedVectors format
        """
        self.model.wv.save_word2vec_format(out_file)

    def load_vectors(self, out_file):
        """
        Load embeddings from gensim.models.KeyedVectors format
        """
        self.model = gensim.wv.load_word2vec_format(out_file)


class SKLearnEmbedder(BaseNodeEmbedder):
    def __init__(self, 
            embedder, 
            normalize_graph=True,
            **kwargs):
        """
        Creates a node embedder through a SKLearn API model
        The SKLearn model must implement the fit_transform method
            (eg. Isomap, UMAP, PCA, etc.)
            
        Model parameters are also passed directly into this constructor
        
        embedder : SKLearn model implementing fit_transform
            This is not a model instance, rather the class name object
            
        normalize_graph : bool
            Whether to normalize the graph edge weights before embedding
            This calls the optimized CSRGraph routine
            
        **kwargs : SKLearn model keyword arguments
            These arguments are passed directly to the construction of the model
        """
        self.embedder = embedder(**kwargs)
        self.normalize_graph = normalize_graph

    def fit(self, graph, verbose=1):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        verbose : bool
            Whether to print output while working
        """
        G = CSRGraph(graph)
        if self.normalize_graph:
            G = G.normalize(return_self=True)
            gc.collect()
        vectors = self.embedder.fit_transform(G.matrix())
        self.model = dict(zip(G.nodes(), vectors))


    def predict(self, node_name):
        """
        Return vector associated with node
        node_name : str or int
            either the node ID or node name depending on graph format
        """
        return self.model[node_name]


class Glove(BaseNodeEmbedder):
    def __init__(self, 
            n_components=32,
            tol=0.001,
            max_epoch=150,
            learning_rate=0.01, 
            max_loss=10.,):
        """
        """
        self.n_components = n_components
        self.tol = tol
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.max_loss = max_loss

    def fit(self, graph, verbose=1):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        verbose : bool
            Whether to print output while working
        """
        G = CSRGraph(graph)
        vectors = G.embeddings(n_components=self.n_components, 
                              tol=self.tol, max_epoch=self.max_epoch,
                              learning_rate=self.learning_rate, 
                              max_loss=self.max_loss,
                              method="edges")
        self.model = dict(zip(G.nodes(), vectors))


    def predict(self, node_name):
        """
        Return vector associated with node
        node_name : str or int
            either the node ID or node name depending on graph format
        """
        return self.model[node_name]
