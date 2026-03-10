# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load the dataset and select relevant features such as Age, Annual Income, and Spending Score.
2. Standardize the selected features to normalize the data.
3. Use the Elbow Method to determine the optimal number of clusters.
4. Apply K-Means clustering, assign cluster labels to the data, and visualize the clusters with centroids.
```

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: LOKESHWARAN G 
RegisterNumber:  212225040210
*/
```
```
import os
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent MKL memory leak warning on Windows

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

# Optional: suppress only the specific warning message if you want cleaner output
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")
# Step 1: Load the dataset
data = pd.read_csv('CustomerData.csv')
# Step 2: Explore the data
print(data.head())
print(data.columns)
# Step 3: Select relevant features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]
# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Step 5: Elbow method to find optimal clusters
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
# Step 6: Train KMeans with chosen clusters
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)  # Explicit n_init
kmeans.fit(X_scaled)
# Step 7: Add cluster labels to data
data['Cluster'] = kmeans.labels_

# Silhouette score
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')
# Step 8: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.7
)
# Plot cluster centroids (inverse scale)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```
## Output:

<img width="881" height="461" alt="Screenshot 2026-03-10 145855" src="https://github.com/user-attachments/assets/06fc4cf1-a33d-4eb8-a85f-a29f32b01141" />

<img width="1115" height="627" alt="Screenshot 2026-03-10 175707" src="https://github.com/user-attachments/assets/3e62b5bf-77e1-4087-a7ff-6013da335d0c" />


## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
