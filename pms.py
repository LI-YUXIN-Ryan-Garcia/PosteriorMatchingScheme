'''
    Normal Posterior Matching Scheme
'''

import numpy as np
from tree import SplayTree

class PMS():
    def __init__(self, error_prob):
        self.errp = error_prob
        self.tree = SplayTree(0, 1, 1)
        left_node = SplayTree(0, 0.5, 0.5)
        self.tree.insert(left_node)
        self.middle = 0.5

    # normal PMS transmission
    def transmit(self, msg): 
        self.msg = msg
        for i in range(500):
            # encoding message
            self.X = 1 if self.msg > self.middle else 0
            # decoding message
            self.Y = 1 - self.X if np.random.rand() < self.errp else self.X

            # update probability
            if self.Y == 0:
                self.tree.left.p, self.tree.right.p = 1 - self.errp, self.errp
            else:
                self.tree.left.p, self.tree.right.p = self.errp, 1 - self.errp

            # find the middle line
            direction = 0 if self.Y == 1 else 1
            middle_node = self.tree.search_node(0.5, direction)
            print("middle: {}".format(middle_node.start_value))
            if abs(self.middle - middle_node.start_value) < 1E-9:
                print("END: {}".format(i))
                return
            self.middle = middle_node.start_value

            self.tree = middle_node.parent.rotate()

