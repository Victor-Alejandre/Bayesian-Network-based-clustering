import networkx as nx
import pandas as pd
import plotly
import pickle
import pybnesian as pb
import numpy as np
import matplotlib.pyplot as plt

#Este código genera una representación de la red con el sombreado de nodos según los valores de importancia introducidos.
#Esto se utiliza para representar la red dado el representante de cada cluster y su importancia

def clusters_dag(red,importance,clusters_names):
    fig1,axes=plt.subplots(1,len(clusters_names))
    for i in range(len(clusters_names)):
        G = nx.DiGraph()
        G.add_nodes_from(red.nodes())
        G.add_edges_from(red.arcs())
        values = [importance[clusters_names[i]].get(node, 0) for node in G.nodes()]

        #pos = nx.planar_layout(G)
        pos = nx.bipartite_layout(G,['cluster'])
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Purples'),ax=axes[i],
                               node_color=values, node_size=2000, vmin=min(values), vmax=max(values), edgecolors='k')
        nx.draw_networkx_labels(G, pos, font_size=12,ax=axes[i])
        nx.draw_networkx_edges(G, pos, arrows=True, node_size=2000,ax=axes[i],connectionstyle='arc3,rad=-0.3')

        axes[i].set_title('cluster'+' '+f"{clusters_names[i]}")

    plt.show()
