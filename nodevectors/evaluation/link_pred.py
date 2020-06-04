import copy
import random
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
     f1_score, roc_auc_score)

def split_train_test_graph(G, testing_ratio=0.5, seed=42):
    node_num1, edge_num1 = len(G.nodes), len(G.edges)
    testing_edges_num = int(len(G.edges) * testing_ratio)
    random.seed(seed)
    testing_pos_edges = random.sample(G.edges, testing_edges_num)
    G_train = copy.deepcopy(G)
    # for edge in testing_pos_edges:
    #     node_u, node_v = edge # unpack edge
    #     if (G_train.degree(node_u) > 1 and G_train.degree(node_v) > 1):
    #         G_train.remove_edge(node_u, node_v)
    G_train.remove_nodes_from(testing_pos_edges)
    G_train.remove_nodes_from(nx.isolates(G_train))
    node_num2, edge_num2 = len(G_train.nodes), len(G_train.edges)
    assert node_num1 == node_num2
    return G_train, testing_pos_edges


def generate_neg_edges(G, testing_edges_num, seed=42):
    nnodes = len(G.nodes)
    # Make a full graph (matrix of 1)
    negG = np.ones((nnodes, nnodes))
    np.fill_diagonal(negG, 0.)
    # Substract existing edges from full graph
    # generates negative graph
    original_graph = nx.adj_matrix(G).todense()
    negG = negG - original_graph
    # get negative edges (nonzero entries)
    neg_edges = np.where(negG > 0)
    random.seed(seed) # replicability!
    rng_edges = random.sample(range(neg_edges[0].size), testing_edges_num)
    # return edges in (src, dst) tuple format
    return list(zip(
        neg_edges[0][rng_edges],
        neg_edges[1][rng_edges]
    ))


def LinkPrediction(embedding, G, train_G, test_pos_edges, seed=42):
    print("Link Prediction:")
    train_neg_edges = generate_neg_edges(G, len(train_G.edges()), seed)
    random.seed(seed)
    # create a auxiliary graph to ensure that testing 
    #    negative edges will not used in training
    G_aux = copy.deepcopy(G)
    G_aux.add_edges_from(train_neg_edges)
    test_neg_edges = generate_neg_edges(G_aux, len(test_pos_edges), seed)

    # construct X_train, y_train, X_test, y_test
    X_train = []
    y_train = []
    for e in train_G.edges():
        feature_vector = np.append(embedding[e[0]], embedding[e[1]])
        X_train.append(feature_vector)
        y_train.append(1)
    for e in train_neg_edges:
        feature_vector = np.append(embedding[e[0]], embedding[e[1]])
        X_train.append(feature_vector)
        y_train.append(0)

    X_test = []
    y_test = []
    for e in test_pos_edges:
        feature_vector = np.append(embedding[e[0]], embedding[e[1]])
        X_test.append(feature_vector)
        y_test.append(1)
    for e in test_neg_edges:
        feature_vector = np.append(embedding[e[0]], embedding[e[1]])
        X_test.append(feature_vector)
        y_test.append(0)

    # shuffle for training and testing
    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    c = list(zip(X_test, y_test))
    random.shuffle(c)
    X_test, y_test = zip(*c)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf1 = LogisticRegression(random_state=seed, solver='lbfgs')
    clf1.fit(X_train, y_train)
    y_pred_proba = clf1.predict_proba(X_test)[:, 1]
    y_pred = clf1.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\t(logit) AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, "
          f"Acc: {accuracy:.3f}, F1: {f1:.3f}")
    # Same as above but with lgbm
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
    clf2.fit(X_train, y_train)
    y_pred_proba = clf2.predict_proba(X_test)[:, 1]
    y_pred = clf2.predict(X_test)
    auc_roc2 = roc_auc_score(y_test, y_pred_proba)
    auc_pr2 = average_precision_score(y_test, y_pred_proba)
    accuracy2 = accuracy_score(y_test, y_pred)
    f12 = f1_score(y_test, y_pred)
    print(f"\t(lgbm)  AUC-ROC: {auc_roc2:.3f}, AUC-PR: {auc_pr2:.3f}, "
          f"Acc: {accuracy2:.3f}, F1: {f12:.3f}")
    return {"auc_roc": auc_roc, 
            "auc_pr" : auc_pr, 
            "accuracy" : accuracy, 
            "f1" : f1}
