import networkx as nx 
import random
import matplotlib.pyplot as plt  
G=nx.Graph()
for u, v in nx.barabasi_albert_graph(10,2,seed=1).edges():
    G.add_edge(u,v,weight=random.uniform(0,0.4))
pos=nx.spring_layout(G,iterations=20)
edgewidth=[]
for (u,v,d) in G.edges(data=True):
     edgewidth.append(round(G.get_edge_data(u,v).values()[0]*20,2))
nx.draw_networkx_nodes(G,pos)
plt.show()