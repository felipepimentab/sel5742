import numpy as np
from utils import matrix as mx

# 1) Training stage
# Loading training data
training_data, r, c = mx.text_to_table('data/epc01/treinamento.txt')
x, d = mx.get_x_and_d(training_data)
# Defining theta
theta = -1.0
# Defining learnig rate
eta = 0.01
# Initiating weights array with small normalized random values
w_i = mx.create_weights_array(c)

w = mx.train_slp(x, d, w_i, eta)
print(f'w = {w}')