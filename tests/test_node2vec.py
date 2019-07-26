import networkx as nx
import numpy as np
import os
from scipy import sparse
import unittest
import warnings

import graph2vec


# This is a markov chain with an absorbing state
# Over long enough, every walk ends stuck at node 1
absorbing_state_graph = sparse.csr_matrix(np.array([
    [1, 0., 0., 0., 0.], # 1 point only to himself
    [0.5, 0.3, 0.2, 0., 0.], # everyone else points to 1
    [0.5, 0, 0.2, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.1, 0.2],
    ], dtype=np.float32)
)

absorbing_state_graph_2 = sparse.csr_matrix(np.array([
    [0.5, 0.5, 0., 0., 0.], # 1 and 2 are an absorbing state
    [0.5, 0.5, 0., 0., 0.], # everyone else can get absorbed
    [0.5, 0.2, 0., 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.1, 0.2],
    ], dtype=np.float32)
)

# This graph is two disconnected subgraphs
# No walk can go from one to the other
disconnected_graph = sparse.csr_matrix(np.array([
    [0.5, 0.5, 0., 0., 0.], # 1 and 2 together
    [0.5, 0.5, 0., 0., 0.],
    [0., 0., 0.7, 0.2, 0.1], # 3-5 together
    [0., 0., 0.1, 0.2, 0.7],
    [0., 0., 0.1, 0.7, 0.2],
    ], dtype=np.float32)
)

class TestGraphEmbedding(unittest.TestCase):
    """
    higher level tests
    """
    def test_wheel_graph(self):
        tt = nx.generators.classic.wheel_graph(100)
        wordsize = 32
        # Gensim triggers deprecation warnings...
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        g2v = graph2vec.Node2Vec(
            walklen=5, epochs=5,
            return_weight=1., 
            neighbor_weight=1., threads=0,
            w2vparams={"window":10, "size":wordsize, "negative":20, "iter":10,
                       "batch_words":128, "workers": 6})
        g2v2 = graph2vec.Node2Vec(
            walklen=5, epochs=5,
            return_weight=1.5, 
            neighbor_weight=0.5, threads=0,
            w2vparams={"window":10, "size":wordsize, "negative":20, "iter":10,
                       "batch_words":128, "workers": 6})
        g2v.fit(tt, verbose=False)
        g2v2.fit(tt, verbose=False)
        self.assertTrue(len(g2v.predict(9)) == wordsize)
        self.assertTrue(len(g2v2.predict(9)) == wordsize)
        warnings.resetwarnings()


class TestNodeWalks(unittest.TestCase):
    """
    Test that Node2Vec walks do as they should
    """
    def test_return_weight_inf_loops(self):
        """
        if return weight ~inf, should loop back and forth
        """
        n_nodes = 5
        n_epoch = 2
        walklen=10
        fully_connected = np.ones((n_nodes,n_nodes))
        np.fill_diagonal(fully_connected, 0)
        fully_connected = graph2vec.graph._sparse_normalize_rows(
            fully_connected
        )
        t1 = graph2vec.graph.make_walks( 
            fully_connected,
            walklen=walklen,
            epochs=n_epoch,
            return_weight=9999,
            neighbor_weight=1.,
            threads=0)
        # Neighbor weight ~ 0 should also loop 
        t2 = graph2vec.graph.make_walks( 
            fully_connected,
            walklen=walklen,
            epochs=n_epoch,
            return_weight=1.,
            neighbor_weight=0.000001,
            # Change thread values to make sure it works
            threads=0)
        self.assertTrue(t1.shape == (n_nodes * n_epoch, walklen))
        # even columns should be equal (always returning)
        np.testing.assert_array_equal(t1[:, 0], t1[:, 2])
        np.testing.assert_array_equal(t1[:, 0], t1[:, 4])
        np.testing.assert_array_equal(t1[:, 0], t1[:, 6])
        np.testing.assert_array_equal(t2[:, 0], t2[:, 2])
        np.testing.assert_array_equal(t2[:, 0], t2[:, 4])
        np.testing.assert_array_equal(t2[:, 0], t2[:, 6])
        # same for odd columns
        np.testing.assert_array_equal(t1[:, 1], t1[:, 3])
        np.testing.assert_array_equal(t1[:, 1], t1[:, 5])
        np.testing.assert_array_equal(t1[:, 1], t1[:, 7])
        np.testing.assert_array_equal(t2[:, 1], t2[:, 3])
        np.testing.assert_array_equal(t2[:, 1], t2[:, 5])
        np.testing.assert_array_equal(t2[:, 1], t2[:, 7])

    def test_no_loop_weights(self):
        """
        if return weight ~inf, should loop back and forth
        """
        n_nodes = 5
        n_epoch = 2
        walklen=10
        fully_connected = np.ones((n_nodes,n_nodes))
        np.fill_diagonal(fully_connected, 0)
        fully_connected = graph2vec.graph._sparse_normalize_rows(
            fully_connected
        )
        t1 = graph2vec.graph.make_walks( 
            fully_connected,
            walklen=walklen,
            epochs=n_epoch,
            return_weight=0.000001,
            neighbor_weight=1.,
            threads=0)
        # Neighbor weight ~ 0 should also loop 
        t2 = graph2vec.graph.make_walks( 
            fully_connected,
            walklen=walklen,
            epochs=n_epoch,
            return_weight=1.,
            neighbor_weight=99999999,
            # Change thread values to make sure it works
            threads=0)
        self.assertTrue(t1.shape == (n_nodes * n_epoch, walklen))

        # Test that it doesn't loop back
        # Difference between skips shouldnt be 0 anywhere
        tres1 = ((t1[:, 0] - t1[:, 2]) != 0)
        tres2 = ((t1[:, 1] - t1[:, 3]) != 0)
        tres3 = ((t1[:, 2] - t1[:, 4]) != 0)
        tres4 = ((t1[:, 3] - t1[:, 5]) != 0)
        for i in [tres1, tres2, tres3, tres4]:
            if not i.all():
                print(f"ERROR in walks\n\n {t1}")
            self.assertTrue(i.all())
        # Second by neighbor weight
        tres1 = ((t2[:, 0] - t2[:, 2]) != 0)
        tres2 = ((t2[:, 1] - t2[:, 3]) != 0)
        tres3 = ((t2[:, 2] - t2[:, 4]) != 0)
        tres4 = ((t2[:, 3] - t2[:, 5]) != 0)
        for i in [tres1, tres2, tres3, tres4]:
            if not i.all():
                print(f"ERROR in walks\n\n {t2}")
            self.assertTrue(i.all())

class TestSparseUtilities(unittest.TestCase):
    """
    Test graph embeddings sub methods
    """
    def test_row_normalizer(self):
        # Multiplying by a constant returns the same
        test1 = disconnected_graph * 3
        test2 = absorbing_state_graph_2 * 6
        test3 = absorbing_state_graph * 99
        # Scipy.sparse uses np.matrix which throws warnings
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        np.testing.assert_array_almost_equal(
            graph2vec.graph._sparse_normalize_rows(test1).toarray(),
            disconnected_graph.toarray(),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            graph2vec.graph._sparse_normalize_rows(test2).toarray(),
            absorbing_state_graph_2.toarray(),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            graph2vec.graph._sparse_normalize_rows(test3).toarray(),
            absorbing_state_graph.toarray(),
            decimal=3
        )
        with self.assertRaises(ValueError):
            graph2vec.graph._sparse_normalize_rows(
                np.array([
                    [1,2,3,4,5,6],
                    [1,0,0,1,0,1],
                    [1,1,1,1,1,1],
                    [0,0,0,0,0,0], # Bad row
                    [1,0,0,0,0,0],
                    [0,10,0,1,0,0.1]
                ])
            )

        warnings.resetwarnings()


    def test_given_disconnected_graph_walks_dont_cross(self):
        walks1 = graph2vec.graph._csr_random_walk(
            disconnected_graph.data,
            disconnected_graph.indptr,
            disconnected_graph.indices,
            np.array([1, 0, 1, 0]), 
            walklen=10
        )
        walks2 = graph2vec.graph._csr_random_walk(
            disconnected_graph.data,
            disconnected_graph.indptr,
            disconnected_graph.indices,
            np.array([2, 4, 3, 2, 4, 3]), 
            walklen=10
        )
        end_state1 = walks1[:, -1]
        end_state2 = walks2[:, -1]
        self.assertTrue(np.isin(end_state1, [0,1]).all(),
            f"Walks: {walks1} \nEndStates: {end_state1}\n"
        )
        self.assertTrue(np.isin(end_state2, [3,4,2]).all(),
            f"Walks: {walks2} \nEndStates: {end_state2}\n"
        )


    def test_random_walk_uniform_dist(self):
        n_nodes = 50
        n_walklen = 100
        fully_connected = graph2vec.graph._sparse_normalize_rows(
            np.ones((n_nodes,n_nodes))
        )
        t1 = graph2vec.graph.make_walks( 
            fully_connected,
            walklen=n_walklen,
            epochs=10,
            threads=0)
        expected_val = (n_nodes - 1) / 2
        self.assertTrue(np.abs(np.mean(t1) - expected_val) < 0.3)


    def test_given_absorbing_graph_walks_absorb(self):
        # Walks should be long enough to avoid flaky tests
        walks1 = graph2vec.graph._csr_random_walk(
            absorbing_state_graph.data,
            absorbing_state_graph.indptr,
            absorbing_state_graph.indices,
            np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]), 
            walklen=80
        )
        walks2 = graph2vec.graph._csr_random_walk(
            absorbing_state_graph_2.data,
            absorbing_state_graph_2.indptr,
            absorbing_state_graph_2.indices,
            np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]), 
            walklen=80
        )
        walks3 = graph2vec.graph.make_walks(
            absorbing_state_graph, 
            walklen=50, 
            epochs=80, 
            threads=0
        )
        walks4 = graph2vec.graph.make_walks(
            absorbing_state_graph, 
            walklen=50, 
            epochs=80, 
            threads=0
        )
        end_state1 = walks1[:, -1]
        end_state2 = walks2[:, -1]
        end_state3 = walks3[:, -1]
        end_state4 = walks4[:, -1]
        self.assertTrue(np.isin(end_state1, [0]).all(),
            f"Walks: {walks1} \nEndStates: {end_state1}\n"
        )
        self.assertTrue(np.isin(end_state2, [0, 1]).all(),
            f"Walks: {walks2} \nEndStates: {end_state2}\n"
        )
        self.assertTrue(np.isin(end_state3, [0]).all(),
            f"Walks: {walks3} \nEndStates: {end_state3}\n"
        )
        self.assertTrue(np.isin(end_state4, [0]).all(),
            f"Walks: {walks4} \nEndStates: {end_state4}\n"
        )

    def test_changing_n_threads_works(self):
        """
        This is the last test, force recompile with different # threads
        """
        walks = graph2vec.graph.make_walks(
            absorbing_state_graph, 
            walklen=50, 
            epochs=80, 
            threads=3 # Different than in other tests
        )
        self.assertTrue(True, "Should get here without issues")
