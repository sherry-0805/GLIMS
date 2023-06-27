import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh


def parse_file_get_index(file):
    index = []
    for line in open(file):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, len):
    mask = np.zeros(len)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess_features(features, sparse=True):
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if sparse:
        return sparse_to_tuple(features)
    else:
        return features.todense()


def preprocess_adj(adj):
    adj_normal = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normal)


def sparse_to_tuple(sparse_matrix):
    def to_tuple(matrix):
        if not sp.isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data
        shape = matrix.shape
        return coords, values, shape

    if isinstance(sparse_matrix, list):
        for i in range(len(sparse_matrix)):
            sparse_matrix[i] = to_tuple(sparse_matrix[i])
    else:
        sparse_matrix = to_tuple(sparse_matrix)

    return sparse_matrix


def normalize_adj(adj, sparse=True):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    if sparse:
        return res.tocoo()
    else:
        return res.todense()


def sub_lower_support(polys):
    for i in range(1, len(polys)):
        for j in range(0, i):
            polys[i][np.abs(polys[j]) > 0.0001] = 0
    return polys


def chebyshev_polynomials(adj, k, sparse=True, sub_support=True):
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normal = normalize_adj(adj, sparse=sparse)
    laplacian = sp.eye(adj.shape[0]) - adj_normal
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    if sparse:
        t_k.append(sp.eye(adj.shape[0]))
    else:
        t_k.append(np.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for _ in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    if sparse:
        if sub_support:
            t_k = sub_lower_support(t_k)
        return sparse_to_tuple(t_k)
    else:
        if sub_support:
            return sub_lower_support(t_k)
        else:
            return t_k


def get_support_matrices(adj, poly_number):
    if poly_number > 0:
        support_matrix = chebyshev_polynomials(adj, poly_number)
        num_supports = 1 + poly_number
    else:
        support_matrix = [sp.eye(adj.shape[0])]
        num_supports = 1
    return support_matrix, num_supports


def construct_feed_dict(expr, mut, cn, methy, support, labels, labels_mask, placeholders):
    """Construct data dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['expr']: expr})
    feed_dict.update({placeholders['mut']: mut})
    feed_dict.update({placeholders['cn']: cn})
    feed_dict.update({placeholders['methy']: methy})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    return feed_dict






