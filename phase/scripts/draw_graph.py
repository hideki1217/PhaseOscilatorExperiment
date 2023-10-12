import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def draw_graph(K, w0, ax, node_size=80, edge_width=1.5):
    K = pd.DataFrame(K, index=w0, columns=w0)
    mask_K = K.mask(np.triu(np.ones(K.shape)).astype(bool), None)
    edge_lists = mask_K.stack().reset_index().apply(tuple, axis=1).values
    G = nx.Graph()
    G.add_weighted_edges_from(edge_lists)

    weights = nx.get_edge_attributes(G, 'weight').values()
    print(weights)
    pos_y = 2 * w0 ** 2
    pos = {w0[i]: (w0[i], pos_y[i]) for i in range(len(w0))}
    nx.draw(G, pos, edge_color=weights,
            edge_cmap=plt.cm.Blues, width=edge_width, node_size=node_size, ax=ax)

    return ax


def save_graph(K, w0, fname, **kwargs):
    fig, ax = plt.subplots()

    draw_graph(K, w0, ax, **kwargs)

    fig.savefig(fname)
    plt.close()
