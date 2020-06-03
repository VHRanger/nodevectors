import io
import gzip
import networkx as nx
import numpy as np
import os
import pandas as pd
import requests
import scipy as sc
from scipy import sparse
import scipy.io
from sklearn import cluster, manifold, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sympy.sandbox.tests.test_indexed_integrals
import sys
import time
import umap
# Silence perf warning
import warnings

sys.path.append(os.path.realpath('..'))

import nodevectors as graph2vec
import csrgraph
from csrgraph import csrgraph
import graph_eval

warnings.simplefilter("ignore")


OUT_FILE = 'weighed_edges.csv'

# G, labels = graph_eval.make_blogcatalog(dedupe=True)
# G, labels = graph_eval.make_cluster_graph(n_nodes=320, n_clusters=18, connections=150, drop_pct=0.2)
G, labels = graph_eval.make_weighed_cluster_graph(n_nodes=500, n_clusters=6, connections=1500, drop_pct=0.2, max_edge_weight=15)
# G, labels = graph_eval.make_snap()

y = labels.label
n_clusters = y.nunique()

X_train, X_test, y_train, y_test = train_test_split(
    labels.node, labels.label, test_size=0.10, 
    random_state=33)



for MAX_LOSS in [25.]:
 for LEARNING_RATE in [0.01]:
  for EMBED_SIZE in [1, 2, 4, 8, 16, 64, 128, 256]:
    try:
      embedder = graph2vec.GGVec(
          n_components=EMBED_SIZE,
          tol="auto",
          tol_samples=50,
          max_epoch=5_000,
          learning_rate=LEARNING_RATE, 
          max_loss=MAX_LOSS,
          threads=0,
          verbose=True
      )
      res = graph_eval.evaluate_embedding(
          embedder, G, labels, n_clusters,
          X_train, X_test, y_train, y_test)
      print(res)
      res = pd.DataFrame([pd.Series(res)])
      res['n_components'] = EMBED_SIZE
      if os.path.isfile(OUT_FILE):
        res.to_csv(OUT_FILE, mode='a', header=False, float_format='%.3f')
      else:
          res.to_csv(OUT_FILE, float_format='%.3f')
    except:
      continue



for MIN_DIST in [0.1, 0.01]:
 for N_NEIGHBORS in [15, 5]:
  for METRIC in ['euclidean', 'cosine']:
    for EMBED_SIZE in [1, 2, 4, 8, 16, 64, 128, 256]:
        try:
            embedder = graph2vec.SKLearnEmbedder(
                umap.UMAP,
                n_neighbors=N_NEIGHBORS,
                min_dist=MIN_DIST,
                metric='euclidean',
                n_components=EMBED_SIZE,
            )
            res = graph_eval.evaluate_embedding(
                embedder, G, labels, n_clusters,
                X_train, X_test, y_train, y_test)
            res['n_components'] = EMBED_SIZE
            print(res)
            res = pd.DataFrame([pd.Series(res)])
            res.to_csv(OUT_FILE, mode='a', header=False)
        except:
            continue



for WALKLEN in [20, 40]: # l in paper
 for EPOCH in [40]: # r in paper
  for N_WEIGHT in [0.3, 1., 3.]:
   for R_WEIGHT in [0.3, 1., 3.]:
    for WINDOW in [10]: # k in paper
     for EMBED_SIZE in [1, 2, 4, 8, 16, 64, 128, 256]: # d in paper
      for NS_EXP in [0.75]: # default, not in paper
       for NEGATIVE in [5]: # default, not in paper
            try:
                embedder = graph2vec.Node2Vec(
                    walklen=WALKLEN,
                    epochs=EPOCH,
                    return_weight=R_WEIGHT,
                    neighbor_weight=N_WEIGHT,
                    n_components=EMBED_SIZE,
                    w2vparams={'window': WINDOW,
                            'negative': NEGATIVE, 
                            'iter': 5,
                            'ns_exponent': NS_EXP,
                            'batch_words': 128}
                )
                res = graph_eval.evaluate_embedding(
                    embedder, G, labels, n_clusters,
                    X_train, X_test, y_train, y_test)
                res = pd.DataFrame([pd.Series(res)])
                res['n_components'] = EMBED_SIZE
                print(res)
                res.to_csv(OUT_FILE, mode='a', header=False, float_format='%.3f')
            except:
                continue














