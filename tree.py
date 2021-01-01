'''
Tree data structure for maintaining probaility 
 - Splay tree
 - AVL tree (NA)

 three functions:
 quantile(p) -> x
 pmf(x) -> p
 pdf(x) -> p (NA)
'''

import bigfloat as bf
import matplotlib.pyplot as plt
from math import isclose

tol = bf.BigFloat(1) / 10**18

class Tree():
    def __init__(self, start_value, length, prob):
        # self.start_value, self.length, self.p = start_value, length, prob
        self.start_value = bf.BigFloat(start_value)
        self.length, self.p = bf.BigFloat(length), bf.BigFloat(prob)
        self.parent, self.left, self.right = None, None, None 

class SplayTree(Tree):
    def __init__(self, start_value, length, prob):
        super().__init__(start_value, length, prob)

    def insert(self, node):
        node.parent = self
        self.left = node
        self.right = SplayTree(node.start_value + node.length, self.length - node.length, 1 - node.p)
        self.right.parent = self

    def quantile(self, probability):
        p = probability
        if not self.left: # leaf
            new_node = SplayTree(self.start_value, self.length * p, p)
            self.insert(new_node)
            return self.right
        else:
            if isclose(self.left.p, p, abs_tol=tol):
                return self.right
            elif self.left.p < p: # the left child's PMF is not enough
                return self.right.quantile( (p - self.left.p) / self.right.p )
            else:
                return self.left.quantile( p / self.left.p )

    def PMF(self, x):
        if isclose(self.start_value + self.length, x, abs_tol=tol):
            return self.p
        elif not self.left: # leaf
            delta_len = x - self.start_value
            if delta_len < 0:
                print("len is negative")
                exit()
            new_node = SplayTree(self.start_value, delta_len, delta_len / self.length)
            self.insert(new_node)
            return self.p * self.left.p
        else:
            # if isclose(self.right.start_value,x,abs_tol=tol):
                # return self.p * self.left.p
            # elif isclose(self.left.start_value,x,abs_tol=tol):
                # return 0.0
            if self.right.start_value < x:
                return self.p * (self.left.p + self.right.PMF(x))             
            else:
                return self.p * self.left.PMF(x)

    def zig(self): # right rotation
        # update probability
        self.left.p *= self.p
        self.right.p *= self.p
        self.p = self.parent.p
        self.parent.p = 1 - self.left.p
        self.right.p /= self.parent.p
        self.parent.right.p = 1 - self.right.p

        # update value and length
        self.start_value = self.parent.start_value # maybe optional
        self.length = self.parent.length
        self.parent.start_value = self.right.start_value
        self.parent.length -= self.left.length
        
        grandparent = self.parent.parent
        # connect right child with parent, disconnect right child
        self.parent.left = self.right
        self.right.parent = self.parent
        self.right = self.parent
        # disconnect parent and grandparent, re-connect its parent
        self.right.parent = self
        self.parent = grandparent
        if grandparent and grandparent.left is self.right:
            grandparent.left = self
        elif grandparent and grandparent.right is self.right:
            grandparent.right = self
        elif not grandparent:
            pass
        else:
            print("Error! Grandparent and parent are not matched in zig!\n {}\n {}\n {}".format(grandparent.start_value, self.left.start_value, self.right.start_value))
            exit()
        return self

    def zag(self): # left rotation
        # update probability
        self.left.p *= self.p
        self.right.p *= self.p
        self.p = self.parent.p
        self.parent.p = 1 - self.right.p
        self.left.p /= self.parent.p
        self.parent.left.p = 1 - self.left.p

        # update value and length
        self.start_value = self.parent.start_value
        self.length = self.parent.length
        self.parent.length -= self.right.length

        grandparent = self.parent.parent
        # connect right child with parent, disconnect left child
        self.parent.right = self.left
        self.left.parent = self.parent
        self.left = self.parent
        # disconnect parent and grandparent, re-connect its parent
        self.left.parent = self
        self.parent = grandparent
        if grandparent and grandparent.left is self.left:
            grandparent.left = self
        elif grandparent and grandparent.right is self.left:
            grandparent.right = self
        elif not grandparent:
            pass
        else:
            print("Error! Grandparent and parent are not matched in zag!\n {}\n {}\n {}".format(grandparent.start_value, self.left.start_value, self.right.start_value))
            exit()
        return self
    
    def rotate(self, subtree=False):
        if not self.parent:
            return self
        if not self.parent.parent:
            if subtree: # root of subtree
                return self
            tmp = self.zig() if self.parent.left is self else self.zag()
            # tmp.left.left.print_node()
            return tmp
            # return self.zig() if self.parent.left is self else self.zag()
        
        grandparent = self.parent.parent
        # grandparent, parent and child are on the same side
        # zig-zig
        if grandparent.left is self.parent and self.parent.left is self:
            return self.parent.zig().left.zig().rotate()
        # zag-zag
        elif grandparent.right is self.parent and self.parent.right is self:
            return self.parent.zag().right.zag().rotate()
        # grandparent, parent and child are on the diff sides
        elif grandparent.left is self.parent and self.parent.right is self:
            return self.zag().zig().rotate()
        elif grandparent.right is self.parent and self.parent.left is self:
            return self.zig().zag().rotate()
        else:
            print("Error! No correction pattern!")
            exit()

    def print_node(self, text='NODE'):
        """ Print out info about node, its children and its parent """
        print('--------------- {} ---------------'.format(text))
        n = self.parent
        if n is not None:
            print('Root: v={} l={} p={}'.format(n.start_value, n.length, n.p))
        else:
            print('Root: {}'.format(n))

        n = self
        if n is not None:
            print('Tree node: v={} l={} p={}'.format(n.start_value, n.length, n.p))
        
        n = self.left
        if n is not None: 
            print('Left child: v={} l={} p={}'.format(n.start_value, n.length, n.p))
        else:
            print('Left child: {}'.format(n))
        
        n = self.right
        if n is not None:
            print('Right child: v={} l={} p={}'.format(n.start_value, n.length, n.p))
        else:
            print('Right child: {}'.format(n))
        print()

    def visualize(self):
        print("-"*80)
        intervals = {'value':[], 'length':[], 'probability':[]}
        self.print_intervals(intervals)
        # print(intervals)
        v, l, p = intervals['value'], intervals['length'], intervals['probability']
        h = [p[i] / l[i] for i in range(len(v))]
        plt.bar(v, h, width=l, align='edge')
        plt.show()
        print("-"*80 + "\n")

    def print_intervals(self, intervals, parent_prob=1):
        if self.left is None: # leaf
            p = parent_prob * self.p
            intervals['value'].append(self.start_value)
            intervals['length'].append(self.length)
            intervals['probability'].append(p)
            # print("[{}, {}]: {}".format(self.start_value, self.start_value+self.length, p))
        else:
            self.left.print_intervals(intervals, parent_prob * self.p)
            self.right.print_intervals(intervals, parent_prob * self.p)

class AVLTree(Tree): # TODO
    def __init__(self, start_value, length, prob):
        super().__init__(start_value, length, prob) # TODO