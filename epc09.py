import numpy as np
import random

# Inicialização da Q-table
Q = {}
actions = [(i, j) for i in range(3) for j in range(3)]

def initialize_Q():
  for i in range(3**9):
    state = np.base_repr(i, base=3).zfill(9)
    Q[state] = {a: 0 for a in actions}

def get_state(board):
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
  return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

def get_next_state(board, action, mark):
  next_board = [row[:] for row in board]
  next_board[action[0]][action[1]] = mark
  return next_board

def train_Q_learning(episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
  initialize_Q()
  for _ in range(episodes):
    board = [[' ' for _ in range(3)] for _ in range(3)]
    state = get_state(board)
    while True:
      if random.uniform(0, 1) < epsilon:
        action = random.choice(get_possible_actions(board))
      else:
        max_Q = max(Q[state], key=Q[state].get)
        action = random.choice([a for a in Q[state] if Q[state][a] == Q[state][max_Q]])
      next_board = get_next_state(board, action, 'X')
      next_state = get_state(next_board)
      
      if is_winner(next_board, 'X'):
        Q[state][action] += alpha * (1 - Q[state][action])
        break
      elif not get_possible_actions(next_board):
        Q[state][action] += alpha * (0 - Q[state][action])
        break
      else:
        opponent_action = random.choice(get_possible_actions(next_board))
        next_board = get_next_state(next_board, opponent_action, 'O')
        next_state = get_state(next_board)
        if is_winner(next_board, 'O'):
          Q[state][action] += alpha * (-1 - Q[state][action])
          break
        else:
          reward = 0
          max_next_Q = max(Q[next_state].values())
          Q[state][action] += alpha * (reward + gamma * max_next_Q - Q[state][action])
          state = next_state

def get_policy():
  policy = {}
  for state in Q:
    policy[state] = max(Q[state], key=Q[state].get)
  return policy

# Treinar o algoritmo de Q-learning
train_Q_learning(10000)

# Obter a política ótima
policy = get_policy()

# Imprimir a política ótima
for state in policy:
  print(f"State: {state} -> Action: {policy[state]}")