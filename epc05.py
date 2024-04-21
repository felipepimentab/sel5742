import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Create directory to save images
if not os.path.exists("imgs/epc05"):
  os.makedirs("imgs/epc05")

data = np.genfromtxt("data/epc05/irisFisher.txt", delimiter='\t', dtype=str)
X_str = data[:, 2:4]  # Petal length and petal width as strings
X = X_str.astype(np.float_)  # Convert strings to float
y = data[:, 4]  # Species


# Step 2: Implement k-NN algorithm
def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1 - x2) ** 2))

def k_nearest_neighbors(X_train, y_train, x_test, k=3):
  distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
  k_indices = np.argsort(distances)[:k]
  k_nearest_labels = [y_train[i] for i in k_indices]
  return k_nearest_labels

def predict_species(X_train, y_train, x_test, k=3):
  k_nearest_labels = k_nearest_neighbors(X_train, y_train, x_test, k)
  species, counts = np.unique(k_nearest_labels, return_counts=True)
  majority_vote = species[np.argmax(counts)]
  return majority_vote

# Step 3: Plot the data
species_mapping = {'versicolor': 0, 'virginica': 1, 'setosa': 2}
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

plt.figure(figsize=(8, 6))
for species, label in species_mapping.items():
  plt.scatter(X[y == species, 0], X[y == species, 1], color=colors[label], marker=markers[label], label=species)

# Plot the new sample
new_sample = np.array([[5.0, 1.45]])
plt.scatter(new_sample[:, 0], new_sample[:, 1], color='black', marker='x', label='New Sample')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Species Classification')
plt.legend()
plt.savefig("imgs/epc05/iris_species_classification.png")

# Step 4: Classify the new sample
new_sample_species = predict_species(X, y, new_sample, k=10)
print("Predicted species of the new sample:", new_sample_species)

# Step 5: Find the 10 closest neighbors
distances = [euclidean_distance(new_sample, x) for x in X]
closest_indices = np.argsort(distances)[:10]
closest_neighbors = [(X[i][0], X[i][1], closest_indices[i], y[i]) for i in range(len(closest_indices))]
print(tabulate(sorted(closest_neighbors, key=lambda d: d[2]), headers=['length', 'width', 'distance', 'especies']))

