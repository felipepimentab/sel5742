import numpy as np

def text_to_table(path: str) -> tuple[np.ndarray, int, int]:
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