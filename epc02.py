import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load training and testing data
train_data = np.loadtxt('data/epc02/treinamento.txt')
test_data = np.loadtxt('data/epc02/teste.txt')

# Extract inputs and outputs
X_train, y_train = train_data[:, :4], train_data[:, 4:]
X_test, y_test = test_data[:, :4], test_data[:, 4:]

best_accuracy = 0
best_error_list = None

# Perform 5 trainings with different initial weights matrices
for i in range(5):
  # Initialize MLP classifier with different initial weights
  mlp = MLPClassifier(
    hidden_layer_sizes=(15,),
    activation='logistic',
    solver='sgd',
    learning_rate_init=0.1,
    max_iter=2000,
    epsilon=0.00001
  )

  print(f"Training {i+1}")

  # Train the MLP
  mlp.fit(X_train, y_train)

  # Test against testing data
  predictions = mlp.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)
  print(f"Accuracy for training {i+1}: {accuracy}")

  # Check if this training has the highest accuracy so far
  if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_error_list = mlp.loss_curve_

# Plot mean quadratic error vs training epoch for the best training
if best_error_list:
  plt.plot(best_error_list)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Quadratic Error')
  plt.title('Mean Quadratic Error vs Training Epoch (Best Training)')
  plt.show()
else:
  print("No training achieved accuracy better than random guessing.")
