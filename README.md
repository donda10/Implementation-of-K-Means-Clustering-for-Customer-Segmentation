# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data – Read the CSV and extract features: Annual Income and Spending Score.
2. Visualize data – Plot a scatter plot of the selected features to understand distribution.
3. Apply K-Means – Initialize KMeans with k=5, fit on the data, and obtain centroids and labels.
4. Visualize clusters – Plot clusters with distinct colors, draw enclosing circles, and mark centroids.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Simon Malachi S
RegisterNumber: 212224040318 
*/
```

```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Mall_Customers.csv")
data
X = data[['Annual Income (k$)' , 'Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score (1-100)")
plt.show()
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids: ")
print(centroids)
print("Label:")
# define colors for each cluster
colors = ['r', 'g', 'b', 'c', 'm']

# plotting the controls
for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')

  #Find minimum enclosing circle
  distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)

  circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)

#Plotting the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='k', label='Centroids')

plt.title('K-means Clustering')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal') 
plt.show()



```

## Output:


![Screenshot 2025-05-19 224710](https://github.com/user-attachments/assets/7f76ceeb-90b3-4088-a159-03b952ed1cb4)


![Screenshot 2025-05-19 224651](https://github.com/user-attachments/assets/a5193b49-f2a6-450d-b743-8def7fcdbfdc)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
