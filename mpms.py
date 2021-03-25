'''
Modified Posterior Matching Scheme with hamming(4,7)

                ------- U1...Un+r  ----------- V1...Vn+r ------
   X1...Xn ---> | Enc | --------> | P = alpha | -------> | Dec |
                -------            -----------           ------  
                                                           |
                            Feedback: Y1...Yn + W          |
                        <----------------------------------|

*** Behaviour of the hamming code decoder
- If no error is detected, the decoder does nothing.
- If detected error(s) can be correct, the decoder tries to recover U even if 
    there are more than one error. 
- If the decoder is unable to correct detected error(s):
    - option 1: use one more bit in feedback, indicating the receiver asks the
        sender to transmit X again. (not very ideal)
    - option 2: pretent the code with error(s) as a valid code and decode it
        regardless of error(s) (not robust when code is long)
'''

import numpy as np
import bigfloat as bf
from tree import SplayTree
from pms import PMS
from hamming import HammingCode
from utility import hamming_err_prob, hamming_LOEP

class MPMS(PMS):
    def __init__(self, crossover_prob, err_prob):
        super().__init__(crossover_prob, err_prob)
        self.peak = 0 # peak value

    # Given a bit seq, return prob' s lower\upper bound it belongs to
    def find_interval(self, y):
        order = int(y,2) # convert to integer
        n = self.msg_len
        if n != len(y):
            print("ERROR! Y has invalid length:{}".format(y))
            exit()
        # prob of intervals to be scaled up
        unit_prob = 1 / 2**n
        lb = bf.BigFloat(order) / 2**n
        ub = bf.BigFloat(order + 1) / 2**n
        return order, lb, ub

    def channel_transmit(self, U, num_err=None):
        a = self.XoverP
        u = U
        n = num_err if num_err is not None else len(u) # number of error
        if n > len(u):
            print("Error! The number of error(s) can't be larger than the length of code")
            print("Length of code: {}, number of error(s): {}".format(len(u), num_err))
            exit()
        positions = np.random.choice(len(u), n, replace=False)
        flags = [True if i in positions else False for i in range(len(u))]

        def flip(x, f):
            p = self.XoverP
            x = int(x)
            if f:
                return 1 - x if np.random.rand() <= p else x
            else:
                return x

        return ''.join([str(elm) for elm in list(map(flip, u, flags))])

    # check transmission end
    def check_ending(self):
        v = self.peak
        # boundaries of decoded real number 
        bin_seq, order = self.real_to_bin(v, len(self.seq))
        # bounaries of pmf
        l = len(bin_seq)
        prob_lower_bound = bf.BigFloat(order) / 2**l
        prob_upper_bound = bf.BigFloat(order+1) / 2**l
        
        p1 = self.tree.PMF(prob_lower_bound)
        p2 = self.tree.PMF(prob_upper_bound)
        return True if p2 - p1 > 1 - self.errP else False

    def transmit(self, seq, max_channel_use=None, err_num=None, msg_len=4):
        # PMS settings
        self.msg_point = self.bin_to_real(seq)
        print("Message: {}, Px: {}".format(self.msg_point, self.XoverP))
        
        # hamming code settings
        self.msg_len = msg_len # hamming code message length
        self.redundant_bits = HammingCode.calc_redundant_bits(msg_len)
        self.block_len = msg_len + self.redundant_bits
        print("Hamming Code ({}, {})".format(self.block_len, msg_len))
        self.undecodable = False #TODO
        # h_err_p = self.XoverP
        # h_err_p = hamming_err_prob(self.XoverP, self.msg_len, self.block_len)
        h_err_p = hamming_err_prob(self.XoverP/2, self.msg_len, self.block_len)
        # estimated by leading order
        # h_err_p = hamming_LOEP(self.XoverP, self.block_len) 
        # h_err_p = 0.1179648

        max_default_use = 500
        MCU = max_channel_use if max_channel_use is not None else max_default_use
        for i in range(MCU):
            # np.random.seed(i) # debug mode
            # split probability tree, figure out which block msg belongs to
            msg_pmf = self.tree.PMF(self.msg_point)
            msg_seq, msg_order = self.real_to_bin(msg_pmf, msg_len)
            self.X = msg_seq
            # print("X: {}".format(self.X))

            # hamming encoding:
            h = HammingCode(self.X)
            U = h.encode() # msg to be send thru channel
            # print("U: {}".format(U))

            # Binary Symmetric Channel transmission:
            v = self.channel_transmit(U, err_num)
            # print("V: {}".format(v))

            # hamming decoding:
            err_pos = h.detectError(v) # reverse order
            if err_pos == 0: # no error
                self.Y = self.X
            elif err_pos <= len(v): # able to recover u from v
                err_pos = len(v) - err_pos
                lost_bit = '1' if v[err_pos] == '0' else '0' # flip the error bit
                correct_v = v[:err_pos] + lost_bit + v[err_pos+1:]
                self.Y = h.decode(correct_v[::-1])
            else: #TODO
                # print("Hamming code can't correct error in {} with error position".format(v, err_pos))
                self.undecodable = True
                # self.Y = h.decode(v) 
                continue    

            # print("Y: {}".format(self.Y))  

            '''
                Update probability: scale up the prob block msg belongs to, and
            scale down the other prob block. Divide tree into three parts by l-
            ower/upper bounds of Y's interval: the left part, the middle part, 
            and the right part. Assume the crossover probability is a, so we s-
            hould scale up P([lb, ub]) by a, and scale down P([0,lb], [ub,1]) 
            by 1 - a. The procedures consist of two steps:
                1. Scale down the left part and scale up the other parts.
                2. Scale up the middle part and scale down the right part.
                
                Note that if either the left part or the right part is empty, 
            the situation is the same as standard posterior matching scheme.
            '''
            # probability lower/upper bounds of Y's interval
            Y_order, Y_pmf_lb, Y_pmf_ub = self.find_interval(self.Y) 
            if Y_order == 0: # left part is empty
                Y_node_ub = self.tree.quantile(Y_pmf_ub)
                self.peak = Y_node_ub.start_value
                self.tree = Y_node_ub.parent.rotate()
                self.tree.left.p, self.tree.right.p = 1 - h_err_p, h_err_p
            elif Y_order == 2**self.msg_len - 1: # right part is empty
                Y_node_lb = self.tree.quantile(Y_pmf_lb)
                self.peak = Y_node_lb.start_value
                self.tree = Y_node_lb.parent.rotate()
                self.tree.left.p, self.tree.right.p = h_err_p, 1 - h_err_p
            else:
                # nodes of lower/upper bounds of Y's interval
                Y_node_lb = self.tree.quantile(Y_pmf_lb) 
                Y_node_ub = self.tree.quantile(Y_pmf_ub)
                # number of intervals in the left\right part
                left_num = Y_order
                right_num = 2**self.msg_len - 1 - Y_order 
                
                unit_prob = bf.div(h_err_p, (2**self.msg_len - 1))
                self.peak = (Y_node_lb.start_value + Y_node_ub.start_value) / 2

                # step 1
                self.tree = Y_node_lb.parent.rotate()
                self.tree.left.p = unit_prob * left_num
                self.tree.right.p = 1 - self.tree.left.p
                
                # step 2
                sub = self.tree.right
                sub.parent = None
                self.tree.right = None
                sub = Y_node_ub.parent.rotate()
                sub_total = 1 - h_err_p + unit_prob * right_num
                sub.left.p = bf.BigFloat(1 - h_err_p) / sub_total
                sub.right.p = 1 - sub.left.p 
                self.tree.right = sub
                sub.parent = self.tree
            # print("-"*80)

            # self.tree.visualize()
         
            if self.check_ending():
                bin_seq, _ = self.real_to_bin(self.peak, len(self.seq))
                return bin_seq, i+1, self.block_len

        bin_seq, _ = self.real_to_bin(self.peak, len(self.seq))
        print("You have reached the maximum expected channel use!")
        return bin_seq, MCU, self.block_len