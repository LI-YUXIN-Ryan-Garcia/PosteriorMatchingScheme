'''
    Normal Posterior Matching Scheme
'''

import numpy as np
import bigfloat as bf
from math import isclose
from tree import SplayTree

class PMS():
    def __init__(self, crossover_prob):
        # channel settings
        self.XoverP = crossover_prob # crossover probability
        self.seq = None

        # posterior probability settings
        self.tree = SplayTree(bf.BigFloat(0), bf.BigFloat(1), bf.BigFloat(1))
        left_node = SplayTree(bf.BigFloat(0), bf.BigFloat(0.5), bf.BigFloat(0.5))
        self.tree.insert(left_node)
        self.middle = bf.BigFloat(0.5)
        
    # encode binary sequence to real number if necessary
    def bin_to_real(self, seq):
        if isinstance(seq, str):
            self.seq = seq
            self.is_bin_seq = True
            # print("L: {}".format(len(seq)))
            return bf.BigFloat(int(seq,2) / 2**(len(seq)))
        else: # seq is a real number 
            return seq
    
    # decode real number to binary sequence
    def real_to_bin(self, n):
        l = len(self.seq)
        decimal = round(n * 2**l)
        return [bin(decimal)[2:], decimal]
    
    # check transmission end
    def check_ending(self, node):
        v = node.start_value
        if self.seq is None:
            if abs(self.middle - v) < bf.BigFloat(1) / 10**18:
                return True
            else:
                return False
        else:
            # boundaries of decoded real number 
            result = self.real_to_bin(v)
            bin_seq, order= result[0], result[1]
            l = len(bin_seq)
            lower_bound = bf.BigFloat(order-0.1) / 2**l
            upper_bound = bf.BigFloat(order+1.1) / 2**l
            # bounaries of posterior probability
            t = self.tree
            p = self.XoverP
            tmp = t.PMF(upper_bound) - t.PMF(lower_bound)
            print("diff PMF: {}".format(tmp))
            # if  tmp != 0 and tmp >= (1 - 0.2):
            if isclose(tmp, 0.9, abs_tol=1E-9):
                return True
            else:
                return False

    # standard PMS transmission
    def transmit(self, msg, rounds=None): 
        self.msg = self.bin_to_real(msg)
        print("Message: {}, Pe: {}".format(self.msg, self.XoverP))
        
        r = rounds if rounds is not None else 1000
        for i in range(r):
            np.random.seed(i)
            # encoding message
            self.X = 1 if self.msg > self.middle else 0
            # decoding message
            self.Y = 1 - self.X if np.random.rand() < self.XoverP else self.X

            # update probability
            if self.Y == 0:
                self.tree.left.p, self.tree.right.p = 1 - self.XoverP, self.XoverP
            else:
                self.tree.left.p, self.tree.right.p = self.XoverP, 1 - self.XoverP

            # find the middle line
            # direction = 0 if self.Y == 1 else 1
            # middle_node = self.tree.search_node(0.5, direction)
            middle_node = self.tree.quantile(0.5, return_node=True)
            print("middle: {}".format(middle_node.start_value))
            print("PMF of middle point {}".format(self.tree.PMF(middle_node.start_value)))
            
            # check ending conditions
            if self.check_ending(middle_node):
                if self.seq is not None:
                    bin_seq = self.real_to_bin(middle_node.start_value)[0]
                    return bin_seq, middle_node.start_value, i+1
                else:
                    return None, middle_node.start_value, i+1
            
            # find the new middle line
            self.middle = middle_node.start_value
            self.tree = middle_node.parent.rotate()

            # self.tree.visualize()
        
        if self.seq is not None:
            bin_seq = self.real_to_bin(middle_node.start_value)[0]
        else:
            bin_seq = None

        if rounds is None:
            print("You have reached the maximum expected transmission rounds!")
            return bin_seq, middle_node.start_value, 1001
        else:
            return bin_seq, middle_node.start_value, rounds

