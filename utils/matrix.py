import numpy as np

type matrix = list[list[float]]

def text_to_table(path: str) -> tuple[matrix, int, int]:
  """
  Reads data from a .txt file and returns it as a matrix
  """
  rows = open(path, 'r').readlines() # reads txt data as an array of its lines
  r = len(rows) # sets the number of rows
  c = len(rows[0].split()) # sets the number of columns
  x = np.zeros((r, c)) # creates a placeholder matrix for "x"

  for i in range(r):
    line = rows[i].split() # creates an array by spliting the line on every whitespace/tab
    for j in range(c):
      x[i][j] = float(line[j]) # assings the corresponding value to the table as a float
  
  return x, r, c

def get_x_and_d(data: matrix, theta = -1.0) -> tuple[matrix, matrix]:
  """
  Extracts input "x" and expected output "d" from training data matrix
  """
  r, c = data.shape # sets the number of rows and columns
  x = np.zeros((r, c)) # creates a placeholder matrix for input "x"
  d = np.zeros((r, 1)) # creates a placeholder matrix for output "d"

  for i in range(r):
    x[i][0] = float(theta) # assigns theta to the first column of "x"
    for j in range(c):
      if j == c-1:
        d[i] = float(data[i][j]) # assigns output values to "d"
      else:
        x[i][j+1] = float(data[i][j]) # assigns input values to "x"

  return x, d