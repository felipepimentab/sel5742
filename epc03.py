import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create directory to save images
if not os.path.exists("imgs/epc03"):
  os.makedirs("imgs/epc03")

# Item 1: Plot data for problem with 2 input attributes and 51 instances

# Load data from "data/epc03/svm1.txt"
data_1 = np.loadtxt("data/epc03/svm1.txt")

# Separate input attributes and class labels for item 1
X_1 = data_1[:, :-1]  # Input attributes
y_1 = data_1[:, -1]   # Class labels

# Separate data points for each class for item 1
X_class1_1 = X_1[y_1 == 1]
X_class2_1 = X_1[y_1 == -1]

# Plot data for item 1
plt.figure()
plt.scatter(X_class1_1[:, 0], X_class1_1[:, 1], marker='+', label='Class +1')
plt.scatter(X_class2_1[:, 0], X_class2_1[:, 1], marker='o', label='Class -1')
plt.xlabel('Attribute 1')
plt.ylabel('Attribute 2')
plt.title('Plot of Data for Item 1')
plt.legend()
plt.savefig("imgs/epc03/item1_plot.png")

# Item 4: Train SVM with C = 1 for problem with 2 input attributes and 51 instances

# Train SVM with C = 1 for item 4
svm_c1_1 = SVC(kernel='linear', C=1)
svm_c1_1.fit(X_1, y_1)

# Plot decision boundary for C = 1 for item 4
plt.figure()
plt.scatter(X_class1_1[:, 0], X_class1_1[:, 1], marker='+', label='Class +1')
plt.scatter(X_class2_1[:, 0], X_class2_1[:, 1], marker='o', label='Class -1')
w_c1_1 = svm_c1_1.coef_[0]
b_c1_1 = svm_c1_1.intercept_[0]
x1_min_1, x1_max_1 = X_1[:, 0].min() - 1, X_1[:, 0].max() + 1
x2_min_1, x2_max_1 = X_1[:, 1].min() - 1, X_1[:, 1].max() + 1
xx1_1, xx2_1 = np.meshgrid(np.arange(x1_min_1, x1_max_1, 0.1), np.arange(x2_min_1, x2_max_1, 0.1))
Z_c1_1 = svm_c1_1.predict(np.c_[xx1_1.ravel(), xx2_1.ravel()])
Z_c1_1 = Z_c1_1.reshape(xx1_1.shape)
plt.contour(xx1_1, xx2_1, Z_c1_1, colors='k', linestyles=['-'], levels=[0], label='Decision Boundary (C=1)')
plt.xlabel('Attribute 1')
plt.ylabel('Attribute 2')
plt.title('Decision Boundary for C = 1 (Item 4)')
plt.legend()
plt.savefig("imgs/epc03/item4_decision_boundary_c1.png")

# Train SVM with C = 100 for item 4
svm_c100_1 = SVC(kernel='linear', C=100)
svm_c100_1.fit(X_1, y_1)

# Plot decision boundary for C = 100 for item 4
plt.figure()
plt.scatter(X_class1_1[:, 0], X_class1_1[:, 1], marker='+', label='Class +1')
plt.scatter(X_class2_1[:, 0], X_class2_1[:, 1], marker='o', label='Class -1')
w_c100_1 = svm_c100_1.coef_[0]
b_c100_1 = svm_c100_1.intercept_[0]
Z_c100_1 = svm_c100_1.predict(np.c_[xx1_1.ravel(), xx2_1.ravel()])
Z_c100_1 = Z_c100_1.reshape(xx1_1.shape)
plt.contour(xx1_1, xx2_1, Z_c100_1, colors='r', linestyles=['--'], levels=[0], label='Decision Boundary (C=100)')
plt.xlabel('Attribute 1')
plt.ylabel('Attribute 2')
plt.title('Decision Boundary for C = 100 (Item 4)')
plt.legend()
plt.savefig("imgs/epc03/item4_decision_boundary_c100.png")

# Item 7: Plot data for problem with 2 input attributes and 750 instances

# Load data from "data/epc03/svm2_treinamento.txt"
data_2_train = np.loadtxt("data/epc03/svm2_treinamento.txt")

# Separate input attributes and class labels for item 7
X_2_train = data_2_train[:, :-1]  # Input attributes
y_2_train = data_2_train[:, -1]   # Class labels

# Separate data points for each class for item 7
X_class1_2_train = X_2_train[y_2_train == 1]
X_class2_2_train = X_2_train[y_2_train == -1]

# Plot data for item 7
plt.figure()
plt.scatter(X_class1_2_train[:, 0], X_class1_2_train[:, 1], marker='+', label='Class +1')
plt.scatter(X_class2_2_train[:, 0], X_class2_2_train[:, 1], marker='o', label='Class -1')
plt.xlabel('Attribute 1')
plt.ylabel('Attribute 2')
plt.title('Plot of Data for Item 7')
plt.legend()
plt.savefig("imgs/epc03/item7_plot.png")

# Item 8: Train five SVM with Gaussian kernel for problem with 2 input attributes and 750 instances

# Train SVMs with Gaussian kernel for item 8
gammas = [0.01, 0.1, 1.0, 10, 100]
accuracies = []

for gamma in gammas:
  svm_gamma = SVC(kernel='rbf', C=1, gamma=gamma)
  svm_gamma.fit(X_2_train, y_2_train)
  
  # Load test data from "data/epc03/svm2_teste.txt"
  data_test = np.loadtxt("data/epc03/svm2_teste.txt")
  X_test = data_test[:, :-1]
  y_test = data_test[:, -1]
  
  # Predict using the trained SVM
  y_pred = svm_gamma.predict(X_test)
  
  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  accuracies.append(accuracy)
  print(f"Accuracy for gamma={gamma}: {accuracy}")

# Plot accuracies for item 8
plt.figure()
plt.plot(gammas, accuracies, marker='o')
plt.xlabel('Gamma')
plt.xscale('log')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different Gamma Values (Item 8)')
plt.savefig("imgs/epc03/item8_accuracy.png")

plt.show()
