import random
import numpy as np
import sys
from operator import itemgetter
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def generate_data(type, number, size, seed):
    """
    generate the six family graphs
    type: which type of graphs going to generate, there are six type: 'Erdos_Renyi' 'Ego' 'Regular' 'Geometric'
          'Power_Law'
          'Barabasi_Albert'
    number: the number of the graphs to generate
    size: the max size of each graph
    """
    ##
    random_size = True
    graph_list = []
    node_num = 0
    graph_size =[]
    #seed = 133
    np.random.seed(seed)
    for i in range(number):
        if random_size == True:
            node_num = np.random.randint(int(0.5*size[0]), size[0])
        else:
            node_num == size[0]
        if type == "Erdos_Renyi":
            p = 0.5
            G = nx.generators.erdos_renyi_graph(node_num, p, seed = seed+i)
            ER_adj = nx.to_numpy_matrix(G)
            graph_list.append(np.asarray(ER_adj))
            graph_size.append(node_num)

        if type == "Ego":
            ## this is implemented by networkx
            m = np.random.uniform(0.3,0.6)
            #G = nx.generators.barabasi_albert_graph(node_num, m, seed= seed+i)
            G = nx.generators.erdos_renyi_graph(node_num, m, seed= seed+i)
            node_and_degree = G.degree()
            #(largest_hub, degree) = sorted(node_and_degree.items(), key=itemgetter(1))[-1]
            (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
            # Create ego graph of main hub
            hub_ego = nx.ego_graph(G, largest_hub)
            ego_adj = nx.to_numpy_matrix(hub_ego)
            graph_list.append(np.asarray(ego_adj))
            graph_size.append(node_num)
        if type == "Regular":
            degree = 4
            G = nx.generators.random_regular_graph(degree, node_num, seed = seed+i)
            regular_adj = nx.to_numpy_matrix(G)
            graph_list.append(np.asarray(regular_adj))
            graph_size.append(node_num)
        if type == "Geometric":
            random.seed(seed+i)
            radius = 2
            p=dict((i,(random.gauss(0,1),random.gauss(0,1))) for i in range(node_num))
            G = nx.random_geometric_graph(node_num, radius,pos=p)
            #radius = 0.5
            #G = nx.generators.random_geometric_graph(node_num, radius)
            geo_adj = nx.to_numpy_matrix(G)
            graph_list.append(np.asarray(geo_adj))
            graph_size.append(node_num)

        if type == "Power_Law":
            gamma = 3
            G = nx.generators.powerlaw_cluster_graph(node_num, gamma,0.5,seed = seed + i)
            #G = nx.generators.random_powerlaw_tree(node_num, gamma, tries=600, seed = seed+i)
            power_adj = nx.to_numpy_matrix(G)
            graph_list.append(np.asarray(power_adj))
            graph_size.append(node_num)

        if type == "Barabasi_Albert":
            edges = 4
            G = nx.generators.barabasi_albert_graph(node_num, edges, seed = seed+i)
            BA_adj = nx.to_numpy_matrix(G)
            graph_list.append(np.asarray(BA_adj))
            graph_size.append(node_num)
    return graph_list, graph_size

if __name__ == "__main__":
    generate_data("Regular", 5, [20 ,20])
