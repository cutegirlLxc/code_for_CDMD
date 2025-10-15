# FAO data analysis

```python
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans 
import itertools as it  
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

Layer_names = pd.read_csv("fao_trade_layers.txt",sep = " ")
multilayer = pd.read_csv("fao_trade_multiplex.txt",sep = " ",header = None)
multilayer.columns = ['layerID', 'node_1', 'node_2', 'weight']
Link = multilayer.drop("weight",axis = 1)
nodes = pd.read_csv("fao_trade_nodes.txt",sep=" ")
nodes = nodes.drop("nodeID",axis = 1)
num = nodes.shape[0]

L = max(Link["layerID"].tolist())
A = np.zeros((L,num,num))


for j in range(Link.shape[0]):
    node1 = Link["node_1"][j]-1
    node2 = Link["node_2"][j]-1
    index = Link["layerID"][j]-1
    A[index,node1,node2] = 1


index=[0, 2, 13, 19, 23, 28, 29, 32, 35, 37,141, 152,158, 206, 292, 294, 301, 302,308, 324]  # Most dense 20 layers 
Tensor_A = np.zeros((20,num,num))
for j in range(20):
    IND = index[j]
    Tensor_A[j] = A[IND]
    
```

```python
adjacency_matrix = Tensor_A[3]

gold_color = (1.0, 215/255, 0.0)  
dark_blue_color = (0, 0, 139/255)  


cmap = LinearSegmentedColormap.from_list('gold_darkblue', [dark_blue_color,gold_color], N=256)


sns.heatmap(adjacency_matrix, annot=False, cmap=cmap, fmt="d", cbar=False)

plt.xticks([])  
plt.yticks([])  
plt.xlabel('Import', fontsize=12)
plt.ylabel('Export', fontsize=12)
plt.show()
```

```python

Sum_Tensor = np.zeros((num,num))
for i in range(20):
    Sum_Tensor += Tensor_A[i]


Row = np.sum(Sum_Tensor, axis = 0)
Col = np.sum(Sum_Tensor, axis = 1)
row_0 = np.where(Row == 0)
col_0 = np.where(Col == 0) 
row_ind = row_0[0].tolist()
num_zero = len(row_ind)
row_num = num-num_zero
Tensor = np.zeros((20,row_num,num))

preserve_node_index = list(set(range(num))-set(row_ind))

node_list = nodes["nodeLabel"].tolist()
preserve_node = []
for j in preserve_node_index:
    preserve_node.append(node_list[j])

preserve_node = pd.DataFrame(preserve_node)

for j in range(20):
    ind = index[j]
    Adj = A[ind]
    Tensor[j] = np.delete(Adj,row_ind,axis=0)


Layernames20 = Layer_names["layerLabel"][index]
Drink = [0,3,6,10,16]
Processed = [0,2,3,6,7,8,11,12,15,16,19]
Crude = [4,5,9,13,14,17,18]
Layernames20

bias_adj = np.zeros((row_num, row_num))
for i in Crude:
    Ti = Tensor[i]
    bias_adj += Ti @ Ti.T  
STAT = bias_adj - np.diag(np.diag(bias_adj))


U, eigen, Vt = np.linalg.svd(STAT)
eigen = eigen.tolist()

Lead_U = U[:,0:4]
U_star  = np.zeros(Lead_U.shape)
for j in range(Lead_U.shape[0]):
    norm = np.linalg.norm(Lead_U[j])
    if norm != 0:
        U_star[j] = Lead_U[j]/norm  

kmeans = KMeans(n_clusters=4,random_state=50).fit(U_star)

k = kmeans.labels_
Group1 = pd.DataFrame(preserve_node)[k==0][0].tolist()
Group2 = pd.DataFrame(preserve_node)[k==1][0].tolist()
Group3 = pd.DataFrame(preserve_node)[k==2][0].tolist()
Group4 = pd.DataFrame(preserve_node)[k==3][0].tolist()
```
