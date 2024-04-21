import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tabulate import tabulate

# Create directory to save images
if not os.path.exists("imgs/epc04"):
  os.makedirs("imgs/epc04")

med_name = ['analgesic', 'anti-inflammatory', 'antibiotic', 'antihistamine', 'antipyretic']

# Item 1:Implement K-means algorithm
data = np.loadtxt('data/epc04/remedios.txt')
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)

# Item 2: Get cluster centers
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
distances = pairwise_distances_argmin_min(data, cluster_centers)[1]

attrs = [None]*k
for i in range(k):
    attrs[i] = [med_name[i], cluster_centers[i][0], cluster_centers[i][1]]
print(tabulate(attrs, headers=['cluster', 'atribute 1 (pH)', 'atribute 2 (sol)']))

# Item 3:
print('\n')
attrs = [None]*k
for i in range(k):
    attrs[i] = [med_name[i], np.sum(labels == i), np.mean(distances[labels == i])]
print(tabulate(attrs, headers=['cluster', 'number of samples', 'mean distance']))

# Item 4: Plot all samples
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c='grey', alpha=0.5)
plt.title('Arrangement of all samples')
plt.xlabel('PH')
plt.ylabel('Solubility')
plt.grid(True)
plt.savefig("imgs/epc04/all_samples.png")

# Item 5: Plot samples with cluster centers
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Group {i+1}')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, c='black', label='Centers')
plt.title('Arrangement of samples with cluster centers')
plt.xlabel('PH')
plt.ylabel('Solubility')
plt.legend()
plt.grid(True)
plt.savefig("imgs/epc04/samples_with_clusters.png")