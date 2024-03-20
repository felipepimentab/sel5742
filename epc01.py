import numpy as np
from utils import matrix as mx

# 1) Training stage
# Loading training data
treinamento, r, c = mx.text_to_table('data/epc01/treinamento.txt')
x, d = mx.get_x_and_d(treinamento)
# Initiating weights array with small normalized random values
w0 = np.random.normal(0, 1, 3)
# Defining learnig rate
eta = 0.01

