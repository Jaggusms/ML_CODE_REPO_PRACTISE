import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import numpy as np
from scipy.spatial import distance_matrix
 
# Create the matrices
# dataset = pd.read_csv('Mall_Customers.csv')
# x = dataset.iloc[:, [3, 4]].values      

class KMeans:
    def __init__(self,n_clusters=2,max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
    def kmeansPP(self,x,K):
        centroids=np.array(object='folat64')
        for k in range(1,K+1):
            if k==1:
                idx=np.random.choice(np.arange(x.shape[0]),1)
                centroids=x[idx]
            elif k==2:
                # compute the distance matrix p=2 ecludian p=1 mahathan 
                dist_mat = distance_matrix(x, x, p=2)
                centroids=x[np.where(dist_mat==dist_mat.max())[0]]
            else:
                max=0.0
                new_point=np.array(object='folat64')
                for point in centroids:
                    try:
                        idx=np.where(np.all(x==point,axis=1).reshape(x.shape[0],-1)==True)[0][0]
                        x=np.delete(x,idx,axis=0)
                    except:
                        pass   
                for row in x:
                    d=0.0
                    for point in centroids:
                        d +=np.sqrt(np.dot(point,row))
                    avg= d/len(centroids)
                    if avg>max:
                        new_point=row
                        max=avg
                centroids=np.concatenate((centroids,[new_point]),axis=0)
        return centroids 
    def fit(self,X):
        self.centroids = self.kmeansPP(X,self.n_clusters)
        for i in range(self.max_iter):
            # assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            # move centroids
            self.centroids = self.move_centroids(X,cluster_group)
            # check finish
            if (old_centroids == self.centroids).all():
                break

        return cluster_group
    def clusters(self):
        return self.centroids

    def assign_clusters(self,X):
        cluster_group = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)

    def move_centroids(self,X,cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))

        return np.array(new_centroids)
    
    def predict_method1(self, x_test):
        centroid_for_test_points=[]
        for row in x_test:
            #print(np.argmin([np.sqrt(np.dot(row-centroid,row-centroid))  for centroid in self.centroids]))
            centroid_for_test_points.append(np.argmin([np.sqrt(np.dot(row-centroid,row-centroid))  for centroid in self.centroids]))
        return centroid_for_test_points
    
    def predict_m2(self, x_test):
        return self.fit(self,x_test)


from sklearn import datasets
X = datasets.load_iris()
input=X.data
target=X.target.reshape(input.shape[0],-1)
#print(np.concatenate((input,target),axis=1))
X=X.data

# dataset = pd.read_csv('Mall_Customers.csv')
# X = dataset.iloc[:, [3, 4]].values
print(X.shape[0])
position=np.random.choice(np.arange(X.shape[0]),int(X.shape[0]*0.8),replace=False)
X_Train=X[position]
print(X_Train.shape[0])
X_test= X[[i for i in range(X.shape[0])if i not in position]]
print(X_test.shape[0])
km=KMeans(n_clusters=3)
train=np.array(km.fit(X_Train)).reshape(X_Train.shape[0],-1)
train_array=np.concatenate((X_Train,train), axis=1)
x_pred=np.array(km.predict_method1(X_test)).reshape(X_test.shape[0],-1)
test_array=np.concatenate((X_test,x_pred), axis=1)
X=np.concatenate((train_array,test_array), axis=0)
# import plotly.express as px
# X=pd.DataFrame(X)
# print(X.columns)
# fig = px.scatter_3d(X, x=X.columns[0], y=X.columns[1], z=X.columns[2],
#               color=X.columns[4])
# fig.show()

y_kmeans=X[:,4].astype(int)
from mpl_toolkits import mplot3d
# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
ax.scatter3D(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter3D(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
ax.scatter3D(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
#plt.scatter, km.clusters, s = 300, c = 'yellow', label = 'Centroids')
#plt.title('Clusters of customers')
#plt.xlabel('Annual Income (k$)')
#plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()