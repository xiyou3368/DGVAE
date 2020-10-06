import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_graph_e(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj
    #adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    true_a = adj_normalized + 0.5 * sym_l.dot(sym_l) - 1/6 * sym_l.dot(sym_l).dot(sym_l)
    return sparse_to_tuple(true_a)

def preprocess_graph_generate(graph_list):
    adj_norms = []
    adj_labels = []

    for idx, adj in enumerate(graph_list):
        ## get the adj_norm and adj_label
        adj_sp = sp.csr_matrix(adj)
        adj_orig = adj_sp - sp.dia_matrix((adj_sp.diagonal()[np.newaxis, :], [0]), shape=adj_sp.shape)
        adj_orig.eliminate_zeros()
        adj = sp.coo_matrix(adj_sp)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        adj_norm = sparse_to_tuple(adj_normalized)
        adj_norms.append(adj_norm)
        ## get adj_label
        adj_label = adj_orig + sp.eye(adj_orig.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_labels.append(adj_label)
    return adj_norms, adj_labels

def preprocess_graph_generate_e(graph_list):
    adj_norms = []
    adj_labels = []

    for idx, adj in enumerate(graph_list):
        ## get the adj_norm and adj_label
        adj_sp = sp.csr_matrix(adj)
        adj_orig = adj_sp - sp.dia_matrix((adj_sp.diagonal()[np.newaxis, :], [0]), shape=adj_sp.shape)
        adj_orig.eliminate_zeros()
        adj = sp.coo_matrix(adj_sp)
        #adj_ = adj + sp.eye(adj.shape[0])
        adj_ = adj
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        ## add another preprocess
        sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
        true_a = adj_normalized + 0.5 * sym_l.dot(sym_l) - 1 / 6 * sym_l.dot(sym_l).dot(sym_l)
        ##################
        adj_norm = sparse_to_tuple(true_a)
        adj_norms.append(adj_norm)
        ## get adj_label
        adj_label = adj_orig + sp.eye(adj_orig.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_labels.append(adj_label)
    return adj_norms, adj_labels

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    #train_edges = edges
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    #assert ~ismember(val_edges, train_edges)
    #assert ~ismember(test_edges, train_edges)
    #assert ~ismember(val_edges, test_edges)
    data = np.ones(train_edges.shape[0])
    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def graph_padding(graph_list, graph_size):
    """padding the graph with the largest size"""
    max_size = max(graph_size)
    for idx, g in enumerate(graph_list):
        temp = np.zeros([max_size, max_size])
        temp[:g.shape[0], :g.shape[1]] = g
        graph_list[idx] = temp
    return graph_list, max_size

def mask_test_graphs(graph_list, graph_size):
    ## mask the test graphs
    graph_list = np.array(graph_list)
    graph_size = np.array(graph_size)
    num_test =max(int(np.floor(len(graph_size) / 3.)),1)
    num_val = max(int(np.floor(len(graph_size) / 3.)),1)
    all_graph_idx = np.array(list(range(len(graph_size))))
    np.random.shuffle(all_graph_idx)
    val_graph_idx = all_graph_idx[:num_val]
    test_graph_idx = all_graph_idx[num_val:(num_val + num_test)]
    test_graph = graph_list[test_graph_idx]
    test_size = graph_size[test_graph_idx]
    val_graph = graph_list[val_graph_idx]
    val_size = graph_size[val_graph_idx]
    # train_edges = edges
    train_graph = np.delete(graph_list, np.hstack([test_graph_idx, val_graph_idx]), axis=0)
    train_size = np.delete(graph_size, np.hstack([test_graph_idx, val_graph_idx]), axis=0)
    return train_graph,train_size, val_graph, val_size, test_graph, test_size

