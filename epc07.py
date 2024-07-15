import math
from collections import Counter

# Data from the table
data = [
  {"Tensão": "Alta", "Temperatura": "Calor", "Humidade": "Alta", "Corrente": "Baixa", "Estado": "Desligado"},
  {"Tensão": "Alta", "Temperatura": "Calor", "Humidade": "Alta", "Corrente": "Alta", "Estado": "Desligado"},
  {"Tensão": "Normal", "Temperatura": "Calor", "Humidade": "Alta", "Corrente": "Baixa", "Estado": "Ligado"},
  {"Tensão": "Baixa", "Temperatura": "Normal", "Humidade": "Alta", "Corrente": "Baixa", "Estado": "Ligado"},
  {"Tensão": "Baixa", "Temperatura": "Frio", "Humidade": "Normal", "Corrente": "Baixa", "Estado": "Ligado"},
  {"Tensão": "Baixa", "Temperatura": "Frio", "Humidade": "Normal", "Corrente": "Alta", "Estado": "Desligado"},
  {"Tensão": "Normal", "Temperatura": "Frio", "Humidade": "Normal", "Corrente": "Alta", "Estado": "Ligado"},
  {"Tensão": "Alta", "Temperatura": "Normal", "Humidade": "Alta", "Corrente": "Baixa", "Estado": "Desligado"},
  {"Tensão": "Alta", "Temperatura": "Frio", "Humidade": "Normal", "Corrente": "Baixa", "Estado": "Ligado"},
  {"Tensão": "Baixa", "Temperatura": "Normal", "Humidade": "Normal", "Corrente": "Baixa", "Estado": "Ligado"},
  {"Tensão": "Alta", "Temperatura": "Normal", "Humidade": "Normal", "Corrente": "Alta", "Estado": "Ligado"},
  {"Tensão": "Normal", "Temperatura": "Normal", "Humidade": "Alta", "Corrente": "Alta", "Estado": "Ligado"},
  {"Tensão": "Normal", "Temperatura": "Calor", "Humidade": "Normal", "Corrente": "Baixa", "Estado": "Ligado"},
  {"Tensão": "Baixa", "Temperatura": "Normal", "Humidade": "Alta", "Corrente": "Alta", "Estado": "Desligado"},
]

def entropy(class_counts):
  total = sum(class_counts.values())
  return -sum((count / total) * math.log2(count / total) for count in class_counts.values() if count > 0)

def information_gain(data, attribute, target_attribute):
  total_entropy = entropy(Counter(item[target_attribute] for item in data))
  subsets = {}
  for item in data:
    key = item[attribute]
    if key not in subsets:
      subsets[key] = []
    subsets[key].append(item)
  subset_entropy = sum((len(subset) / len(data)) * entropy(Counter(item[target_attribute] for item in subset)) for subset in subsets.values())
  return total_entropy - subset_entropy

def build_decision_tree(data, attributes, target_attribute):
  # Calculate the initial entropy
  class_counts = Counter(item[target_attribute] for item in data)
  initial_entropy = entropy(class_counts)

  # Calculate information gain for each attribute
  info_gains = {attr: information_gain(data, attr, target_attribute) for attr in attributes}

  # Select the attribute with the highest information gain
  best_attr = max(info_gains, key=info_gains.get)
  tree = {best_attr: {}}

  # Split the dataset based on the best attribute
  subsets = {}
  for item in data:
    key = item[best_attr]
    if key not in subsets:
      subsets[key] = []
    subsets[key].append(item)

  # Recursively build the tree for each subset
  remaining_attributes = [attr for attr in attributes if attr != best_attr]
  for key, subset in subsets.items():
    if len(set(item[target_attribute] for item in subset)) == 1:
      tree[best_attr][key] = subset[0][target_attribute]
    else:
      tree[best_attr][key] = build_decision_tree(subset, remaining_attributes, target_attribute)

  return tree

# Attributes and target attribute
attributes = ["Tensão", "Temperatura", "Humidade", "Corrente"]
target_attribute = "Estado"

# Calculate initial entropy and information gain for each attribute
class_counts = Counter(item[target_attribute] for item in data)
initial_entropy = entropy(class_counts)
info_gains = {attr: information_gain(data, attr, target_attribute) for attr in attributes}

# Build the decision tree
decision_tree = build_decision_tree(data, attributes, target_attribute)

# Print the initial entropy and information gain for each attribute
print(f"Initial Entropy: {initial_entropy}")
for attr, gain in info_gains.items():
  print(f"Information Gain for {attr}: {gain}")

# Print the decision tree
print("Decision Tree:")
print(decision_tree)