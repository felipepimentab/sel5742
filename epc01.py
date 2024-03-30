from utils import slp
from utils import file 

# 1) Training stage
# Loading training data
training_data, r, c = file.text_to_table('data/epc01/treinamento.txt')
x_train, d = slp.get_x_and_d(training_data)
# Defining theta for trainig
theta = -1.0
# Defining learnig rate for training
eta = 0.01
# Defining number of trainings 
T = 5
# Creating empty weights arrays (inital and final)
w_i, w_f = {}, {}

for i in range(1, T+1):
  # Initiating weights array with small normalized random values
  w_i["T{i}"] = slp.create_weights_array(c)
  print(f'w_i[T{i}] = {w_i["T{i}"]}')
  # Training perceptron / adjusting weights
  w_f["T{i}"] = slp.train(x_train, d, w_i["T{i}"], eta)
  print(f'w_f[T{i}] = {w_f["T{i}"]}')
  print("-------------------------------------------------------------")

# 2) Testing stage
# Loading testing data
testing_data, rt, ct = file.text_to_table('data/epc01/teste.txt')
x_test = slp.get_x(testing_data)
# Creating empty results array
y = {}
for i in range(1, T+1):
  # Testing results with trained perceptron / adjusted weights
  y["T{i}"] = slp.test(x_test, w_f["T{i}"])
  print(f"y[T{i}] = {y["T{i}"].T}")

# 3) Changing the treshold
# Creating new weights array with theta = +1
w_i["TH"] = slp.create_weights_array(c, 1.0)
print(f'w_i[TH] = {w_i["TH"]}')
w_f["TH"] = slp.train(x_train, d, w_i["TH"], eta)
print(f'w_f[TH] = {w_f["TH"]}')
y["TH"] = slp.test(x_test, w_f["TH"])
print(f"y[TH] = {y["TH"].T}")