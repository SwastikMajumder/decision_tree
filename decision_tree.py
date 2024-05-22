from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

import math

def sub_key_entropy(col, item, outcome): # if this is a leaf node
    a = list(zip(col, outcome)).count((item, "1"))
    b = list(zip(col, outcome)).count((item, "0"))
    if a == 0:
        return "0"
    elif b == 0:
        return "1"
    else:
        return None

def key_entropy(col, outcome): # find entropy
    unique = list(set(col))
    entropy_list = []
    for item in unique:
        a = list(zip(col, outcome)).count((item, "1"))
        b = list(zip(col, outcome)).count((item, "0"))
        entropy_list.append(S(a,b)*(a+b)/len(col))
    outcome_entropy = S(outcome.count("1"), outcome.count("0"))
    return outcome_entropy - sum(entropy_list)

def max_entropy(table, outcome): # find a root node
    max_list = []
    for key in table.keys():
        max_list.append((key, key_entropy(table[key], outcome)))
    max_list = sorted(max_list, key=lambda x: x[1])
    return max_list[-1]

import copy

def split_table(table, key, item, outcome): # split
    outcome_copy = outcome.copy()
    table_copy = copy.deepcopy(table)
    col = table_copy[key].copy()
    for i in range(len(col)-1,-1,-1):
        if col[i]!=item:
            for inner_key in table_copy.keys():
                table_copy[inner_key].pop(i)
            outcome_copy.pop(i)
    del table_copy[key]
    return table_copy, outcome_copy

def S(a, b):
    if a == 0 or b == 0:
        return 0
    return -(a/(a+b))*math.log2(a/(a+b)) - (b/(a+b))*math.log2(b/(a+b))

class Node:
    def __init__(self, label, children):
        self.label = label
        self.children = children
        
def build_tree(table, outcome): # the function which build the decision tree
    output = max_entropy(table, outcome)
    tree = Node(output[0], {})
    arr = list(set(table[output[0]]))
    for item in arr:
        sub_output = sub_key_entropy(table[output[0]], item, outcome)
        if sub_output is None:
            edit_table, edit_outcome = split_table(table, output[0], item, outcome)
            tree.children[item] = build_tree(edit_table, edit_outcome) # recursion
        else:
            tree.children[item] = sub_output # tree ends at leaf node
    return tree

def predict(tree, data_point):
    for item in data_point:
        if item[0] == tree.label and item[1] in tree.children.keys():
            val = tree.children[item[1]]
            if isinstance(val, str):
                return val
            else:
                return predict(val, data_point)
            
for _ in range(100):
    content = None
    with open("house_votes_84.csv", "r") as file:
        content = file.read()
    content = content.split("\n")
    content = content[1:]
    content = np.array(content)
    content = shuffle(content)
    content = train_test_split(content, test_size=0.2)
    train, test = content
    train_f = np.array([np.array(item.split(",")[:-1]).astype(str) for item in train])
    test_f = np.array([np.array(item.split(",")[:-1]).astype(str) for item in test])
    train_c = [item.split(",")[-1] for item in train]
    test_c = [item.split(",")[-1] for item in test]

    train_f = train_f.transpose()

    orig_table = {}
    for i in range(len(train_f)):
        orig_table[str(i)] = list(train_f[i])

    orig_outcome = train_c

    output = build_tree(orig_table, orig_outcome)

    correct = 0 # training data on testing data
    for i in range(len(test_f)):
        data_point = [(str(j), test_f[i][j]) for j in range(len(test_f[i]))]
        if predict(output, data_point) == test_c[i]:
            correct += 1
    print(correct/len(test_f))

    train_f = train_f.transpose()
    correct = 0 # training data on training data
    for i in range(len(train_f)):
        data_point = [(str(j), train_f[i][j]) for j in range(len(train_f[i]))]
        if predict(output, data_point) == train_c[i]:
            correct += 1
    print(correct/len(train_f))
    print()
