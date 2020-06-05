import io
import gzip
import networkx as nx
import numpy as np
import pandas as pd
import random
import requests
import scipy as sc
import scipy.io
from sklearn import cluster, manifold, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import time
import umap
import warnings

import csrgraph as cg

def make_blogcatalog(edgelist="../data/blogcatalog.mat",
                    dedupe=True):
    """
    Graph with cluster labels from blogcatalog
    
    Dedupe: Whether to deduplicate results (else some nodes have multilabels)
    """
    mat = scipy.io.loadmat(edgelist)
    nodes = mat['network'].tocsr()
    groups = mat['group']
    G = nx.from_scipy_sparse_matrix(nodes)
    labels = (
        pd.DataFrame(groups.todense())
        .idxmax(axis=1)
        .reset_index(drop=False)
    )
    labels.columns = ['node', 'label']
    labels.node = labels.node.astype(int)
    if dedupe:
        labels = labels.loc[~labels.node.duplicated()
                        ].reset_index(drop=True)
        labels.label = labels.label.astype(int) - 1
        return G, labels
    else:
        df = pd.DataFrame(groups.todense())
        labels_list = df.apply(lambda row: list((row.loc[row > 0]).index), axis=1)
        return G, pd.DataFrame({'node': list(G), 'mlabels': pd.Series(labels_list)})

def get_karateclub(graph_name):
    """
    Gets formatted dataset from KarateClub library
    https://karateclub.readthedocs.io
    """
    try:
        from karateclub import GraphReader
    except:
        raise Exception(
            "get_karateclub requires the karateclub library!\n"
            + "Try 'pip install karateclub'\n"
            + "see https://github.com/benedekrozemberczki/KarateClub")
    G = GraphReader(graph_name).get_graph()
    y = GraphReader(graph_name).get_target()
    return G, pd.DataFrame({'node': list(G), 'label': pd.Series(y)})

def make_email():
    """
    Graph from university emails, clustered by departments
    Data from http://snap.stanford.edu/data/email-Eu-core.html
    Edge list Format
    """
    res = requests.get('http://snap.stanford.edu/data/email-Eu-core.txt.gz', verify=False)
    edges = gzip.GzipFile(fileobj=io.BytesIO(res.content))
    edges = pd.read_csv(io.StringIO(edges.read().decode()), header=None, sep=' ')
    edges.columns = ['src', 'dest']
    # cluster labels per node
    res = requests.get('http://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz', verify=False)
    labels = gzip.GzipFile(fileobj=io.BytesIO(res.content))
    labels = pd.read_csv(io.StringIO(labels.read().decode()), header=None, sep=' ')
    labels.columns = ['node', 'cluster']
    G = nx.Graph()
    G.add_edges_from([(t.src, t.dest) for t in edges.itertuples()])
    return G, pd.DataFrame({'node': list(G), 'label': labels.cluster})

def get_from_snap(url="http://snap.stanford.edu/data/wiki-Vote.txt.gz", 
                  sep='\t', header=None, comment='#'):
    """
    Download graph from SNAP dataset
    """
    res = requests.get(url, verify=False)
    edges = gzip.GzipFile(fileobj=io.BytesIO(res.content))
    edges = pd.read_csv(io.StringIO(edges.read().decode()), 
                        header=header, sep=sep, comment=comment)
    edges.columns = ['src', 'dest']
    G = nx.Graph()
    G.add_edges_from([(t.src, t.dest) for t in edges.itertuples()])
    return G

#############################
#                           #
#        RNG DATASETS       #
#                           #
#############################

def make_cluster_graph(
        n_nodes, n_clusters, 
        connections=1, drop_pct=0.1, 
        max_edge_weight=None):
    """
    Makes distinct complete subgraphs
        connected by random paths
        
    n_nodes (int): number of nodes
    n_clusters (int): number of clusters
        This is also the number of disjoint subgraphs
    connections (int): number of random connections 
        These join the disjoint subgraphs
    """
    div = int(n_nodes / n_clusters)
    subgraph_sizes = [div] * n_clusters
    # last cluster has remainder nodes
    subgraph_sizes[-1] = subgraph_sizes[-1] + (n_nodes % n_clusters)
    # Make G from disjoint subgraphs
    G = nx.complete_graph(subgraph_sizes[0])
    for i in range(1, len(subgraph_sizes)):
        G = nx.disjoint_union(G, nx.complete_graph(subgraph_sizes[i]))
    # connecting paths
    for i in range(connections):
        while True:
            c1, c2 = np.random.randint(n_nodes, size=2)
            if G.has_edge(c1, c2):
                continue
            G.add_edge(c1, c2)
            break
    # Drop random edges
    n_edges = len(G.edges)
    to_remove=random.sample(G.edges(),
                            k=int(n_edges * drop_pct))
    G.remove_edges_from(to_remove)
    # Generate labels
    labels = []
    for i in range(len(subgraph_sizes)):
        labels.append([i] * subgraph_sizes[i])
    labels = sum(labels, [])
    assert len(labels) == n_nodes, f"{labels}"
    assert len(set(labels)) == n_clusters, f"{labels}"
    return G, pd.DataFrame({'node': list(G), 'label': pd.Series(labels)})


def make_weighed_cluster_graph(
        n_nodes, n_clusters, 
        connections=1, drop_pct=0.1, 
        max_edge_weight=10):
    """
    Makes distinct complete subgraphs
        connected by random paths
        
    n_nodes (int): number of nodes
    n_clusters (int): number of clusters
        This is also the number of disjoint subgraphs
    connections (int): number of random connections 
        These join the disjoint subgraphs
    """
    div = int(n_nodes / n_clusters)
    subgraph_sizes = [div] * n_clusters
    # last cluster has remainder nodes
    subgraph_sizes[-1] = subgraph_sizes[-1] + (n_nodes % n_clusters)
    # Make G from disjoint subgraphs
    G = nx.complete_graph(subgraph_sizes[0])
    for i in range(1, len(subgraph_sizes)):
        G = nx.disjoint_union(G, nx.complete_graph(subgraph_sizes[i]))
    for eg in G.edges:
        G[eg[0]][eg[1]]['weight'] = np.random.randint(0, max_edge_weight)
    # connecting paths
    for i in range(connections):
        while True:
            c1, c2 = np.random.randint(n_nodes, size=2)
            if G.has_edge(c1, c2):
                continue
            G.add_edge(c1, c2)
            G[c1][c2]['weight'] = np.random.randint(0, max_edge_weight)
            break
    # Drop random edges
    n_edges = len(G.edges)
    to_remove=random.sample(G.edges(),
                            k=int(n_edges * drop_pct))
    G.remove_edges_from(to_remove)
    # Generate labels
    labels = []
    for i in range(len(subgraph_sizes)):
        labels.append([i] * subgraph_sizes[i])
    labels = sum(labels, [])
    assert len(labels) == n_nodes, f"{labels}"
    assert len(set(labels)) == n_clusters, f"{labels}"
    return G, pd.DataFrame({'node': list(G), 'label': pd.Series(labels)})

#############################
#                           #
#        BIO DATASETS       #
#                           #
#############################

def read_bionev_labels(filename):
    """
    Reads multilabels in BioNEV format
    eg. node label1 label2 ... labeln
    ex.
        1 5 8 99 103
        2 4
        3 9 192 777
    Returns pd.DataFrame with nodeID
    """
    fin = open(filename, 'r')
    node_list = []
    labels = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        node_list.append(int(vec[0]))
        labels.append([int(x) for x in vec[1:]])
    fin.close()
    res = pd.DataFrame({'nodes': node_list, 'mlabels':pd.Series(labels)})
    res = res.sort_values(by='nodes').reset_index(drop=True)
    res.nodes = res.nodes - 1
    if not (res.nodes == res.index).all():
        warnings.warn("There are some nodes without multilabels!")
    return res

def get_n2v_ppi(dir_name="../data/bioNEV/node2vec_PPI"):
    """
    Node2vec PPI dataset from BioNEV
    """
    G = cg.read_edgelist(dir_name + "/node2vec_PPI.edgelist", sep=' ')
    labels = read_bionev_labels(dir_name + "/node2vec_PPI_labels.txt")
    G = nx.Graph(G.mat)
    return G, labels

def get_drugbank_ddi(dir_name="../data/bioNEV/DrugBank_DDI"):
    """
    DrugBank DDI dataset from BioNEV
    """
    G = cg.read_edgelist(dir_name + "/DrugBank_DDI.edgelist", sep=' ')
    G = nx.Graph(G.mat)
    return G

def get_mashup_ppi(dir_name="../data/bioNEV/Mashup_PPI"):
    """
    DrugBank DDI dataset from BioNEV
    """
    G = cg.read_edgelist(dir_name + "/Mashup_PPI.edgelist", sep=' ')
    labels = read_bionev_labels(dir_name + "/Mashup_PPI_labels.txt")
    G = nx.Graph(G.mat)
    return G, labels

#############################
#                           #
#         EVALUATION        #
#                           #
#############################


def evalClusteringOnLabels(clusters, groupLabels, verbose=True):
    """
    Evaluates clustering against labels
    Alternative methodology to label prediction for testing
    """
    results = []
    results.append(metrics.adjusted_mutual_info_score(clusters, groupLabels))
    results.append(metrics.adjusted_rand_score(clusters, groupLabels))
    results.append(metrics.fowlkes_mallows_score(clusters, groupLabels))
    if verbose:
        print(f"MI: {results[0]:.2f}, RAND {results[2]:.2f}, FM: {results[2]:.2f}")
    return dict(zip(['MI', 'RAND', 'FM'], np.array(results)))

def get_y_pred(y_test, y_pred_prob):
    """
    Map probabilities to multilabel predictions
    Assuming pre-knowledge of how many classes a node has
    From BioNEV paper
        https://github.com/xiangyue9607/BioNEV
    """
    y_pred = np.zeros(y_pred_prob.shape)
    sort_index = np.flip(np.argsort(y_pred_prob, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = np.sum(y_test[i])
        for j in range(num):
            y_pred[i][sort_index[i][j]] = 1
    return y_pred

def LabelPrediction(w, y, test_size=0.2, seed=42):
    """
    Prints Label Predictions for Labeled Graphs
    Works with Multilabel and single label graphs
    """
    print("Label Prediction:")
    X_train, X_test, y_train, y_test = train_test_split(w, y, 
                                                        test_size=test_size, 
                                                        random_state=seed)
    model = OneVsRestClassifier(linear_model.LogisticRegression(solver='lbfgs',
                                                                random_state=seed))
    model.fit(X_train, y_train)
    if len(y.shape) > 1:
        y_pred_prob = model.predict_proba(X_test)
        ### Assume we know how many label to predict
        y_pred = get_y_pred(y_test, y_pred_prob)
    else:
        y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    micro_f1 = metrics.f1_score(y_test, y_pred, average="micro")
    macro_f1 = metrics.f1_score(y_test, y_pred, average="macro")
    print(f"\t(logit) Acc: {accuracy:.3f}, F1 micro: {micro_f1:.3f}"
          f", F1 macro: {micro_f1:.3f}")
    import lightgbm as lgbm
    clf2 = lgbm.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=15,
        max_depth=3,
        learning_rate=0.25,
        n_estimators=50,
        subsample_for_bin=30,
        objective='mse',
        min_split_gain=0.001,
        min_child_weight=0.001,
        min_child_samples=15,
        importance_type='gain',
    )
    model = OneVsRestClassifier(clf2)
    model.fit(X_train, y_train)
    if len(y.shape) > 1:
        y_pred_prob = model.predict_proba(X_test)
        ### Assume we know how many label to predict
        y_pred = get_y_pred(y_test, y_pred_prob)
    else:
        y_pred = model.predict(X_test)
    accuracy_lgb = metrics.accuracy_score(y_test, y_pred)
    micro_f1_lgb = metrics.f1_score(y_test, y_pred, average="micro")
    macro_f1_lgb = metrics.f1_score(y_test, y_pred, average="macro")
    print(f"\t(lgbm) Acc: {accuracy_lgb:.3f}, F1 micro: {micro_f1_lgb:.3f}"
          f", F1 macro: {micro_f1_lgb:.3f}")
    return {
      "accuracy" : accuracy,
      "micro_f1" : micro_f1,
      "macro_f1" : macro_f1,
      "accuracy_lgb" : accuracy_lgb,
      "micro_f1_lgb" : micro_f1_lgb,
      "macro_f1_lgb" : macro_f1_lgb,
    }

def print_labeled_tests(w, y, test_size=0.2, seed=42):
    """
    Clustering and label prediction tests
    """
    X_train, X_test, y_train, y_test = train_test_split(
        w, y, test_size=test_size, random_state=seed)
    # Print Label Prediction Tests
    res = LabelPrediction(w, y, test_size=test_size, seed=seed)
    # Can only cluster on single-label (not multioutput)
    if len(y.shape) < 2:
        n_clusters = np.unique(y).size
        umpagglo = cluster.AgglomerativeClustering(
            n_clusters=n_clusters, 
            affinity='cosine', 
            linkage='average'
        ).fit(w).labels_
        x = evalClusteringOnLabels(umpagglo, y, verbose=True)
        res = {**res, **x}
    return res
