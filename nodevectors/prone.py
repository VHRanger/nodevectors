import numpy as np
import scipy
from scipy import sparse, linalg
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

from nodevectors.embedders import BaseNodeEmbedder
import csrgraph as cg

class ProNE(BaseNodeEmbedder):
    def __init__(self, n_components=32, step=10, mu=0.2, theta=0.5, 
                exponent=0.75, verbose=True):
        """
        Fast first order, global method.

        Embeds by doing spectral propagation over an initial SVD embedding.
        This can be seen as augmented spectral propagation.

        Parameters :
        --------------
        step : int >= 1
            Step of recursion in post processing step.
            More means a more refined embedding.
            Generally 5-10 is enough
        mu : float
            Damping factor on optimization post-processing
            You rarely have to change it
        theta : float
            Bessel function parameter in Chebyshev polynomial approximation
            You rarely have to change it
        exponent : float in [0, 1]
            Exponent on negative sampling
            You rarely have to change it
        References:
        --------------
        Reference impl: https://github.com/THUDM/ProNE
        Reference Paper: https://www.ijcai.org/Proceedings/2019/0594.pdf
        """
        self.n_components = n_components
        self.step = step
        self.mu = mu
        self.theta = theta
        self.exponent = exponent
        self.verbose = verbose

    
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
        features_matrix = self.pre_factorization(G.mat,
                                                 self.n_components,
                                                 self.exponent)
        vectors = ProNE.chebyshev_gaussian(
            G.mat, features_matrix, self.n_components,
            step=self.step, mu=self.mu, theta=self.theta)
        self.model = dict(zip(G.nodes(), vectors))
        return vectors

    
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
        features_matrix = self.pre_factorization(G.mat,
                                                 self.n_components, 
                                                 self.exponent)
        vectors = ProNE.chebyshev_gaussian(
            G.mat, features_matrix, self.n_components,
            step=self.step, mu=self.mu, theta=self.theta)
        self.model = dict(zip(G.nodes(), vectors))

    @staticmethod
    def tsvd_rand(matrix, n_components):
        """
        Sparse randomized tSVD for fast embedding
        """
        l = matrix.shape[0]
        # Is this csc conversion necessary?
        smat = sparse.csc_matrix(matrix)
        U, Sigma, VT = randomized_svd(smat, 
            n_components=n_components, 
            n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        return U

    @staticmethod
    def pre_factorization(G, n_components, exponent):
        """
        Network Embedding as Sparse Matrix Factorization
        """
        C1 = preprocessing.normalize(G, "l1")
        # Prepare negative samples
        neg = np.array(C1.sum(axis=0))[0] ** exponent
        neg = neg / neg.sum()
        neg = sparse.diags(neg, format="csr")
        neg = G.dot(neg)
        # Set negative elements to 1 -> 0 when log
        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1
        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)
        C1 -= neg
        features_matrix = ProNE.tsvd_rand(C1, n_components=n_components)
        return features_matrix

    @staticmethod
    def svd_dense(matrix, dimension):
        """
        dense embedding via linalg SVD
        """
        U, s, Vh = linalg.svd(matrix, full_matrices=False, 
                              check_finite=False, 
                              overwrite_a=True)
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        return U

    @staticmethod
    def chebyshev_gaussian(G, a, n_components=32, step=10, 
                           mu=0.5, theta=0.5):
        """
        NE Enhancement via Spectral Propagation

        G : Graph (csr graph matrix)
        a : features matrix from tSVD
        mu : damping factor
        theta : bessel function parameter
        """
        nnodes = G.shape[0]
        if step == 1:
            return a
        A = sparse.eye(nnodes) + G
        DA = preprocessing.normalize(A, norm='l1')
        # L is graph laplacian
        L = sparse.eye(nnodes) - DA
        M = L - mu * sparse.eye(nnodes)
        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a
        conv = scipy.special.iv(0, theta) * Lx0
        conv -= 2 * scipy.special.iv(1, theta) * Lx1
        # Use Bessel function to get Chebyshev polynomials
        for i in range(2, step):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * scipy.special.iv(i, theta) * Lx2
            else:
                conv -= 2 * scipy.special.iv(i, theta) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
        mm = A.dot(a - conv)
        emb = ProNE.svd_dense(mm, n_components)
        return emb