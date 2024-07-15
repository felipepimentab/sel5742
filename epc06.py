import numpy as np

# The dataset from Table 1
data = [
  ['True', 'False', 'False', 'True', 'Positive'],
  ['True', 'False', 'False', 'False', 'Positive'],
  ['False', 'True', 'False', 'False', 'Negative'],
  ['False', 'False', 'True', 'False', 'Negative'],
  ['True', 'True', 'False', 'False', 'Positive']
]

# Initialize the most specific hypothesis for S (specific boundary)
S = ['0', '0', '0', '0']

# Initialize the most general hypothesis for G (general boundary)
G = [['?', '?', '?', '?']]

def is_consistent(hypothesis, example):
  for i, val in enumerate(hypothesis):
    if val != '?' and val != example[i]:
      return False
  return True

def update_S(S, example):
  for i in range(len(S)):
    if S[i] == '0':
      S[i] = example[i]
    elif S[i] != example[i]:
      S[i] = '?'
  return S

def update_G(G, S, example):
  G_new = []
  for hypothesis in G:
    if not is_consistent(hypothesis, example):
      for i in range(len(hypothesis)):
        if hypothesis[i] == '?':
          new_hypothesis = hypothesis[:]
          new_hypothesis[i] = S[i]
          if is_consistent(new_hypothesis, example):
            G_new.append(new_hypothesis)
    else:
      G_new.append(hypothesis)
  return [hyp for hyp in G_new if is_consistent(S, hyp)]

# Version Space Algorithm
def version_space(data):
  S = ['0', '0', '0', '0']
  G = [['?', '?', '?', '?']]

  for example in data:
    attributes = example[:-1]
    classification = example[-1]
    if classification == 'Positive':
      S = update_S(S, attributes)
      G = [g for g in G if is_consistent(g, attributes)]
    else:
      G = update_G(G, S, attributes)

  return S, G

# Find-S Algorithm
def find_S(data):
  S = ['0', '0', '0', '0']

  for example in data:
    attributes = example[:-1]
    classification = example[-1]
    if classification == 'Positive':
      S = update_S(S, attributes)

  return S

S, G = version_space(data)
S_find_s = find_S(data)

print("Version Space Method:")
print(f"Specific Boundary (S): {S}")
print(f"General Boundary (G): {G}")

print("\nFind-S Algorithm:")
print(f"Hypothesis: {S_find_s}")
