import numpy as np

import matplotlib.pyplot as plt

# Posterior matching scheme

class Tree:
    parent, left, right = None, None, None

    def __init__(self, start_value, length, p):
        self.start_value = start_value
        self.length = length
        self.p = p
    
    def insert(self, node):
        self.left = node
        node.parent = self
        self.right = Tree(self.start_value + node.length, self.length - node.length, 1 - node.p)
        self.right.parent = self

# EDITED 2
def update(node, middle, value_0, value_1):
    if not node.left:
        if node.start_value + node.length / 2 < middle:
            node.p *= value_0
        else:
            node.p *= value_1
        return
    
    # if the node lies in the left part of the middle line
    # all the nodes at the left of the node are in the range of 0
    elif node.right.start_value < middle:
        node.left.p *= value_0
        update(node.right, middle, value_0, value_1)
    
    # if the node lies in the right part of the middle line
    # all the nodes at the right of the node are in the range of 1
    else:
        node.right.p *= value_1
        update(node.left, middle, value_0, value_1)
        
    # normalize node.p
    sum_p = node.left.p + node.right.p
    node.p *= sum_p
    node.left.p /= sum_p
    node.right.p /= sum_p
    
def find_middle(node, p, direction): # complexity: O(log n)
    if not node.left:
        # direction determines if any interval is changed from 0 to 1 or from 1 to 0
        if direction == 0:
            new_node = Tree(node.start_value, p * node.length, p)
            node.insert(new_node)
        else:
            new_node = Tree(node.start_value, node.length - p * node.length, 1 - p)
            node.insert(new_node)
        return node.right
    
    # when received symbol is 1 (the probability of 1 increase)
    if direction == 0:
        # the remaining probability should be divided first from the left side
        if p < node.left.p:
            return find_middle(node.left, p / node.left.p, direction)
        # if the probability of the left side is not enough, divide from the right side
        else:
            return find_middle(node.right, (p - node.left.p) / node.right.p, direction)
    
    # when received symbol is 0 (the probability of 0 increase)
    else:
        # the remaining probability should be divided first from the right side
        if p < node.right.p:
            return find_middle(node.right, p / node.right.p, direction)
        # if the probability of the right side is not enough, divide from the left side
        else:
            return find_middle(node.left, (p - node.right.p) / node.left.p, direction)

def rotate(node):
    # print("node info {}, {}, {}".format(node.start_value, node.length, node.p))
    # the first situation: single rotate (include checking if the parent node is the root)
    def zig(node):
        if node.parent.left == node:
            #print("left")

            # update the probability, start value and length of the nodes if necessary
            # update two children's probability to the upper level
            node.left.p *= node.p
            node.right.p *= node.p
            # exchange information of the node and its parent
            node.p, node.start_value, node.length = node.parent.p, node.parent.start_value, node.parent.length
            node.parent.p, node.parent.length = 1 - node.left.p, node.length - node.left.length
            # update the new probability of the two children
            node.right.p /= node.parent.p
            node.parent.right.p = 1 - node.right.p

            # rotate the nodes
            # build the connection between the right child and the node's parent
            node.parent.left = node.right
            node.right.parent = node.parent
            # change the connection between the node and its parent
            node.right = node.parent
            tmp = node.parent
            node.parent = node.parent.parent
            tmp.parent = node
            # check if the grandparent of the origin node exists; if exists, update the parent information as well
            if node.parent and node.parent.left == node.right:
                node.parent.left = node
            if node.parent and node.parent.right == node.right:
                node.parent.right = node
    
        elif node.parent.right == node: # similar above (with the opposite direction)
            node.right.p *= node.p
            node.left.p *= node.p
            node.p, node.start_value, node.length = node.parent.p, node.parent.start_value, node.parent.length
            node.parent.p, node.parent.length = 1 - node.right.p, node.length - node.right.length
            node.left.p /= node.parent.p
            node.parent.left.p = 1 - node.left.p

            node.parent.right = node.left
            node.left.parent = node.parent
            node.left = node.parent
            tmp = node.parent
            node.parent = node.parent.parent
            tmp.parent = node
            if node.parent and node.parent.left == node.left:
                node.parent.left = node
            if node.parent and node.parent.right == node.left:
                node.parent.right = node
        
        return node
    
    if node.parent == None:
        return node   
    elif node.parent == initial_node:
        return zig(node)
    
    # the second situation: node and its parent node are both right or left children
    elif node == node.parent.left and node.parent == node.parent.parent.left:
        return rotate(zig(zig(node.parent).left)) 
    elif node == node.parent.right and node.parent == node.parent.parent.right:
        return rotate(zig(zig(node.parent).right))
    
    # the third situation: node and its parent node are children of different sides
    elif node == node.parent.left and node.parent == node.parent.parent.right:
        return rotate(zig(zig(node)))
    elif node == node.parent.right and node.parent == node.parent.parent.left:
        return rotate(zig(zig(node)))


if __name__ == "__main__":
    # initial setting
    m = 0.6
    error_probability = 0.2

    initial_node = Tree(0, 1, 1)
    left_node = Tree(0, 0.5, 0.5)
    initial_node.insert(left_node)
    middle = 0.5

    for i in range(100):
        # encode message m
        if m > middle:
            x = 1
        else:
            x = 0
        print("x = {}".format(x))
        
        # EDITED
        if np.random.rand() < error_probability:
            y = 1 - x
        else:
            y = x
        print("y = {}".format(y))

        # update probability
        if y == 0:
            initial_node.left.p, initial_node.right.p = 1 - error_probability, error_probability
        if y == 1:
            initial_node.left.p, initial_node.right.p = error_probability, 1 - error_probability

        # find the new division line
        if y == 0:
            node = find_middle(initial_node, 0.5, 1)
        else:
            node = find_middle(initial_node, 0.5, 0)
        
        middle = node.start_value
        print(middle)

        initial_node = rotate(node.parent)