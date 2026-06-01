# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Initialize the Dataset Load the customer dataset and select the required features such as Annual Income and Spending Score.

Choose the Number of Clusters (K) Decide the number of clusters to form using methods like the Elbow Method.

Assign Data Points to Clusters Calculate the distance between each data point and cluster centroid, then assign each point to the nearest centroid.

Update Cluster Centroids Recalculate the centroid of each cluster by taking the mean of all data points assigned to it.

Repeat Until Convergence Continue assigning points and updating centroids until the centroids no longer change significantly and final customer segments are formed.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Shyam v
RegisterNumber:  212224220102
*/



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Gender': ['Male','Female','Female','Male','Female','Male','Male','Female','Female','Male'],
    'Age': [19,21,20,23,31,22,35,30,25,28],
    'Annual Income (k$)': [15,16,17,18,19,20,21,22,23,24],
    'Spending Score (1-100)': [39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)

df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8,6))

for i in range(3):
    plt.scatter(
        X[df['Cluster']==i]['Annual Income (k$)'],
        X[df['Cluster']==i]['Spending Score (1-100)'],
        label=f'Cluster {i+1}'
    )

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    c='yellow',
    label='Centroids',
    marker='X'
)

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print(df)
```

## Output:

<img width="846" height="565" alt="image" src="https://github.com/user-attachments/assets/77ab0949-b88b-4e3f-af45-cfd3947a84da" />

<img width="669" height="427" alt="image" src="https://github.com/user-attachments/assets/05aa47de-90c0-4d0a-b1d7-4d169fad9bf9" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
