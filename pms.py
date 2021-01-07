'''
    Normal Posterior Matching Scheme
'''

import numpy as np
import bigfloat as bf
from tree import SplayTree

class PMS():
    def __init__(self, crossover_prob, err_prob):
        # channel settings
        self.XoverP = crossover_prob # crossover probability
        self.errP = err_prob # error probability
        self.seq = None

        # probability tree settings
        self.tree = SplayTree(bf.BigFloat(0), bf.BigFloat(1), bf.BigFloat(1))
        left_node = SplayTree(bf.BigFloat(0), bf.BigFloat(0.5), bf.BigFloat(0.5))
        self.tree.insert(left_node)
        self.peak = bf.BigFloat(0.5)
        
    # encode binary sequence to real number
    def bin_to_real(self, seq):
        self.seq = seq
        dec = int(seq, 2)
        lb = bf.div(dec, 2**(len(seq)))
        ub = bf.div(dec+1, 2**(len(seq)))
        return (lb+ub)/2
    
    # decode real number to binary sequence
    def real_to_bin(self, num):
        l = len(self.seq)
        decimal = int(round(num * 2**l,2)) # 0.49999 => 0.5, 0.48999 => 0.49
        if decimal >= 2**l: # TODO potential error in MPMS
            decimal = 2**l - 1
        return format(decimal, str(l)+'b').replace(' ','0'), decimal
    
    # check transmission terminal
    def check_ending(self, node):
        v = node.start_value
        # boundaries of decoded real number 
        bin_seq, order = self.real_to_bin(v)
        # bounaries of pmf
        l = len(bin_seq)
        prob_lower_bound = bf.BigFloat(order) / 2**l
        prob_upper_bound = bf.BigFloat(order+1) / 2**l
        
        p1 = self.tree.PMF(prob_lower_bound)
        p2 = self.tree.PMF(prob_upper_bound)
        return True if p2 - p1 > bf.sub(1, self.errP) else False

    # standard PMS transmission
    def transmit(self, seq, max_channel_use=None): 
        self.msg = self.bin_to_real(seq)
        # print("Message: {}, Px: {}".format(self.msg, self.XoverP))
        
        max_default_use = 500
        MCU = max_channel_use if max_channel_use is not None else max_default_use
        for i in range(MCU):
            # np.random.seed(i) # debug mode
            # encoding message
            self.X = 1 if self.msg > self.peak else 0
            # decoding message
            self.Y = 1 - self.X if np.random.rand() < self.XoverP else self.X

            # update probability
            if self.Y == 0:
                self.tree.left.p, self.tree.right.p = 1 - self.XoverP, self.XoverP
            else:
                self.tree.left.p, self.tree.right.p = self.XoverP, 1 - self.XoverP

            # find the new middle point
            middle_node = self.tree.quantile(0.5)
            self.peak = middle_node.start_value
            # print("middle: {}".format(self.peak)) # debug mode

            self.tree = middle_node.parent.rotate()

            # check ending conditions
            if self.check_ending(middle_node):
                # self.tree.visualize() #not useful when intervals are too tiny
                bin_seq, _ = self.real_to_bin(middle_node.start_value)
                return bin_seq, middle_node.start_value, i+1

        bin_seq, _ = self.real_to_bin(self.peak)
        print("You have reached the maximum expected channel uses!")
        return bin_seq, self.peak, MCU



