import numpy as np
from utils import matrix as mx

# 1) Training stage
# Loading training data
training_data, r, c = mx.text_to_table('data/epc01/treinamento.txt')
x_train, d = mx.get_x_and_d(training_data)
# Defining theta for trainig
theta = -1.0
# Defining learnig rate for training
eta = 0.01
# Defining number of trainings 
T = 5
# Creating empty weights arrays (inital and final)
w_i, w_f = {}, {}

for i in range(1, T):
  # Initiating weights array with small normalized random values
  w_i["T{i}"] = mx.create_weights_array(c)
  print(f'w_i[T{i}] = {w_i["T{i}"]}')
  # Training perceptron / adjusting weights
  w_f["T{i}"] = mx.train_slp(x_train, d, w_i["T{i}"], eta)
  print(f'w_f[T{i}] = {w_f["T{i}"]}')
  print("-------------------------------------------------------------")

# 2) Testing stage
# Loading testing data
testing_data, rt, ct = mx.text_to_table('data/epc01/teste.txt')
x_test = mx.get_x(testing_data)
# Creating empty results array
y = {}
for i in range(1, T):
  # Testing results with trained perceptron / adjusted weights
  y["T{i}"] = mx.test_slp(x_test, w_f["T{i}"])
  print(f"y[T{i}] = {y["T{i}"].T}")

# 3) Changing the treshold
# Creating new weights array with theta = +1
w_i["TH"] = mx.create_weights_array(c, 1.0)
print(f'w_i[TH] = {w_i["TH"]}')
w_f["TH"] = mx.train_slp(x_train, d, w_i["TH"], eta)
print(f'w_f[TH] = {w_f["TH"]}')
y["TH"] = mx.test_slp(x_test, w_f["TH"])
print(f"y[TH] = {y["TH"].T}")