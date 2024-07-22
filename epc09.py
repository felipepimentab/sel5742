import numpy as np
import random

# Initialize the Q-table with small random values
Q = {}
actions = [(i, j) for i in range(3) for j in range(3)]

def initialize_Q():
  """
  Initialize the Q-table with all possible states.
  Each state is represented as a base-3 string of length 9,
  and each action in the state is initialized to a small random value.
  """
  for i in range(3**9):
    state = np.base_repr(i, base=3).zfill(9)
    Q[state] = {a: np.random.uniform(-1, 1) for a in actions}

def get_state(board):
  """
  Convert the 3x3 Tic-Tac-Toe board into a string representation.
  
  Args:
  board (list of list): Current board state
  
  Returns:
  str: State representation as a string
  """
  state = ''
  for row in board:
    for cell in row:
      if cell == 'X':
        state += '1'
      elif cell == 'O':
        state += '2'
      else:
        state += '0'
  return state

def is_winner(board, mark):
  """
  Check if the given mark ('X' or 'O') has won the game.
  
  Args:
  board (list of list): Current board state
  mark (str): The mark to check for a win ('X' or 'O')
  
  Returns:
  bool: True if the mark has won, False otherwise
  """
  for row in board:
    if all([cell == mark for cell in row]):
      return True
  for col in range(3):
    if all([row[col] == mark for row in board]):
      return True
  if all([board[i][i] == mark for i in range(3)]) or all([board[i][2-i] == mark for i in range(3)]):
    return True
  return False

def get_possible_actions(board):
  """
  Get all possible actions (empty cells) on the board.
  
  Args:
  board (list of list): Current board state
  
  Returns:
  list of tuple: List of possible actions (coordinates of empty cells)
  """
  return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

def get_next_state(board, action, mark):
  """
  Get the next board state after applying the given action.
  
  Args:
  board (list of list): Current board state
  action (tuple): The action to apply (row, column)
  mark (str): The mark to place ('X' or 'O')
  
  Returns:
  list of list: The new board state after the action
  """
  next_board = [row[:] for row in board]
  next_board[action[0]][action[1]] = mark
  return next_board

def train_Q_learning(episodes, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=100):
  """
  Train the Q-learning algorithm for Tic-Tac-Toe.
  
  Args:
  episodes (int): Number of training episodes
  alpha (float): Learning rate
  gamma (float): Discount factor
  epsilon (float): Exploration rate
  max_steps (int): Maximum steps per episode
  """
  initialize_Q()
  for _ in range(episodes):
    board = [[' ' for _ in range(3)] for _ in range(3)]
    state = get_state(board)
    steps = 0
    while steps < max_steps:
      steps += 1
      # Epsilon-greedy action selection
      if random.uniform(0, 1) < epsilon:
        action = random.choice(get_possible_actions(board))
      else:
        max_Q = max(Q[state], key=Q[state].get)
        action = random.choice([a for a in Q[state] if Q[state][a] == Q[state][max_Q]])
      
      # Apply the action and get the next state
      next_board = get_next_state(board, action, 'X')
      next_state = get_state(next_board)
      
      # Check if 'X' wins
      if is_winner(next_board, 'X'):
        Q[state][action] += alpha * (1 - Q[state][action])
        break
      elif not get_possible_actions(next_board):  # Check for draw
        Q[state][action] += alpha * (0 - Q[state][action])
        break
      else:
        # Opponent's move
        opponent_action = random.choice(get_possible_actions(next_board))
        next_board = get_next_state(next_board, opponent_action, 'O')
        next_state = get_state(next_board)
        
        # Check if 'O' wins
        if is_winner(next_board, 'O'):
          Q[state][action] += alpha * (-1 - Q[state][action])
          break
        elif not get_possible_actions(next_board):  # Check for draw
          Q[state][action] += alpha * (0 - Q[state][action])
          break
        else:
          reward = 0
          max_next_Q = max(Q[next_state].values())
          Q[state][action] += alpha * (reward + gamma * max_next_Q - Q[state][action])
          state = next_state

def get_policy():
  """
  Extract the optimal policy from the Q-table.
  
  Returns:
  dict: Optimal policy mapping states to actions
  """
  policy = {}
  for state in Q:
    policy[state] = max(Q[state], key=Q[state].get)
  return policy

# Train the Q-learning algorithm
train_Q_learning(10000)

# Get the optimal policy
policy = get_policy()

# Write the optimal policy to a file
with open('optimal_policy.txt', 'w') as file:
  for state in policy:
    file.write(f"State: {state} -> Action: {policy[state]}\n")

print("Optimal policy has been written to 'optimal_policy.txt'")