import numpy as np

type matrix = list[list[float]]

def step_bipolar(u, treshold = 0):
  if u<treshold:
    return -1
  else:
    return 1

def text_to_table(path: str) -> tuple[matrix, int, int]:
  """
  Reads training data from a .txt file and returns it as a matrix
  """
  rows = open(path, 'r').readlines() # reads txt data as an array of its lines
  k = len(rows) # sets the number of rows (samples)
  n = len(rows[0].split()) # sets the number of columns (inputs)
  x = np.zeros((k, n)) # creates a placeholder matrix for "x"

  for i in range(k):
    line = rows[i].split() # creates an array by spliting the line on every whitespace/tab
    for j in range(n):
      x[i][j] = float(line[j]) # assings the corresponding value to the table as a float
  
  return x, k, n

def get_x(data: matrix, x_0 = -1.0):
  """
  Extracts input samples "x" from testing data matrix
  """
  k, n = data.shape
  x = np.zeros((k, n+1))

  for i in range(k):
    x[i][0] = float(x_0)
    for j in range(n):
      x[i][j+1] = float(data[i][j])

  return x

def get_x_and_d(data: matrix, x_0 = -1.0) -> tuple[matrix, matrix]:
  """
  Extracts input samples "x" and expected output "d" from training data matrix
  """
  k, n = data.shape # sets the number of samples and number of inputs
  x = np.zeros((k, n)) # creates a placeholder matrix for input samples "x"
  d = np.zeros((k, 1)) # creates a placeholder matrix for output "d"

  for i in range(k):
    x[i][0] = float(x_0) # assigns x_0 to the first column of "x"
    for j in range(n):
      if j == n-1:
        d[i] = float(data[i][j]) # assigns output values to "d"
      else:
        x[i][j+1] = float(data[i][j]) # assigns input values to "x"

  return x, d

def create_weights_array(size: int, theta: float = -1.0):
  """
  Creates a normaly distributed wheights array with "theta" as "w[0]"
  """
  w = 2*np.random.random((1,size)) - 1 
  w[0][0] = theta

  return w

def train_slp(x, d, w_i, eta, max_epoch = 10000):
  """
  Training a Single-Layer Perceptron
  """
  k, n = x.shape # defines number of samples (k) and number of inputs (n)
  error = True
  w = w_i
  epoch = 0

  while error == True:
    error = False # FALSE

    for i in range(k):
      u = np.dot(w, x[i])
      y = step_bipolar(u)

      if y!=d[i]:
        w = w + eta * (d[i] - y) * x[i]
        error = True

    epoch += 1
    if epoch >= max_epoch:
      error = False

  print(f'Training finished in {epoch} epochs.')
  
  return w

def test_slp(x, w_f):
  k, n = x.shape # defines number of samples (k) and number of inputs (n)
  w = w_f
  y = np.zeros((k,1))

  for i in range(k):
    u = np.dot(w, x[i])
    y[i] = step_bipolar(u)

  return y