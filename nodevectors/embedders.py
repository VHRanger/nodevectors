import gc
import joblib
import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import shutil
from sklearn.base import BaseEstimator
import sklearn
import sys
import tempfile
import time
import warnings

import csrgraph as cg

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
            "csrg_": cg.__version__[:-2]
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
            csrv = cg.__version__[:-2]
            if meta_data["csrg_"] != csrv:
                raise UserWarning(
                    "Invalid csrgraph version; {0}, required: {1}".format(
                        csrv, meta_data["csrg_"]))
        return model

    def predict(self, node_name):
        """
        Return vector associated with node
        node_name : str or int
            either the node ID or node name depending on graph format
        """
        return self.model[node_name]

    def predict_new(self, 
            neighbor_list,
            pooling=lambda x : np.mean(x, axis=0)):
        """
        Predicts a new node from its neighbors
        Generally done through the average of the embeddings
            of its neighboring nodes.
            
        neighbor_list : iterable[node_id or node name]
            list of node names/IDs with edges to the new node to be predicted
        
        pooling : function array[ndim] -> array[1d]
            function to reduce the embeddings. Normally np.mean or np.sum
            This takes the embeddings of the neighbors and reduces them 
               to the final estimate for the unseen node

        TODO: test
        """
        return pooling([self.predict(x) for x in neighbor_list])


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
            This calls the optimized csrgraph routine
            
        **kwargs : SKLearn model keyword arguments
            These arguments are passed directly to the construction of the model
        """
        self.embedder = embedder(**kwargs)
        self.normalize_graph = normalize_graph

    def fit(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by csrgraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph)
        if self.normalize_graph:
            G = G.normalize(return_self=False)
            gc.collect()
        vectors = self.embedder.fit_transform(G.mat)
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
        if self.normalize_graph:
            G = G.normalize(return_self=True)
            gc.collect()
        vectors = self.embedder.fit_transform(G.mat)
        self.model = dict(zip(G.nodes(), vectors))
        return vectors
