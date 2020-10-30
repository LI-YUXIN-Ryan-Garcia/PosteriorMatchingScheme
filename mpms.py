'''
Modified Posterior Matching Scheme with hamming(4,7)

                ------- U1...Un+r  ----------- V1...Vn+r ------
   X1...Xn ---> | Enc | --------> | P = alpha | -------> | Dec |
                -------            -----------           ------  
                                                           |
                            Feedback: Y1...Yn              |
                        <----------------------------------|
'''

import numpy as np
from tree import SplayTree
from pms import PMS
from hamming import HammingCode

class MPMS(PMS):
    def __init__(self, error_prob, n):
        self.n = n
        super().__init__(error_prob)
        self.value = 0

    # return lower bound and upper bound of the prob block to be scaled up
    def find_interval(self, y):
        order = int(y,2)
        # print("# {} block to be scaled up".format(order))
        n = self.n
        # prob of intervals to be scaled down
        unit_prob = 1 / 2**n
        lb = order * unit_prob
        ub = lb + unit_prob
        return lb, ub

    def channel_transmit(self, U, num_err=1):
        a = self.errp
        u = U
        n = num_err # number of error
        if num_err > len(u):
            print("Error! The number of error(s) can't be larger than the length of code")
            print("Length of code: {}, number of error(s): {}".format(len(u), num_err))
            exit()
        positions = np.random.choice(len(u), n, replace=False)
        flags = [True if i in positions else False for i in range(len(u))]

        def flip(x, f):
            p = self.errp
            x = int(x)
            if f:
                return 1 - x if np.random.rand() < p else x
            else:
                return x

        return ''.join([str(elm) for elm in list(map(flip, u,flags))])

    def transmit(self, msg, numErr=1):
        self.msg = msg
        n = self.n
        target = 0 # target interval
        while True:
            if target / 2**n <= msg and (target+1) / 2**n >= msg:
                break
            target += 1
        print('Target intervel: {}'.format(target))

        for i in range(40):
            
            # encoding message
            self.X = format(target, '{}b'.format(n)).replace(' ', '0')
            print("X: {}".format(self.X))
            
            # hamming encoding:
            h = HammingCode(self.X)
            u = h.genCode() # msg to be send
            print('Sending U_1:n: {}'.format(u)) 

            # channel transmission:
            v = self.channel_transmit(u, numErr)
            print('Receive V_1:n: {}'.format(v))

            # hamming decoding:
            err_pos = h.detectError(v) # reverse order
            if err_pos == 0: # no error
                self.Y = self.X
            else: # recover u from v
                err_pos = len(v) - err_pos
                lost_bit = 1 if v[err_pos] == '0' else 0 # flip the error bit
                correct_v = v[:err_pos] + str(lost_bit) + v[err_pos+1:] 
                print("Correct code: {}".format(correct_v))
                self.Y = h.decode(correct_v[::-1])
            
            print("Y: {}".format(self.Y))

            # update probability, scale up the prob block msg belongs to, and 
            # scale down the other prob block.
            p_lb, p_ub = self.find_interval(self.Y) # prob lower/upper bound
            print("lower bound {}, upper bound: {}".format(p_lb, p_ub))

            '''
                Divide tree into three parts by lower bound and upper bound: 
            the left part, the middle part, and the right part. Assume the err
            prob is a, so we should scale up P([lb, ub]) by a, and scale down
            P([0,lb], [ub,1]) by 1 - a. The procedures consist of two steps:
                1. Scale down the left part and scale up the other parts.
                2. Scale up the middle part and scale down the right part.
                
                If either the left part or the right part is empty, the situa-
            tion is the same as normal posterior matching scheme.
            '''
            tmp = self.tree.left.p
            if p_lb == 0: # 1st block, similar to normal PMS
                node_ub = self.tree.search_node(p_ub, 0)
                self.value = node_ub.start_value
                self.tree = node_ub.parent.rotate()
                self.tree.left.p, self.tree.right.p = 1 - self.errp, self.errp
            elif p_ub == 1: # last blck, similar to normal PMS
                node_lb = self.tree.search_node(p_lb, 0)
                self.value = node_lb.start_value
                self.tree = node_lb.parent.rotate()
                self.tree.left.p, self.tree.right.p = self.errp, 1 - self.errp
            else:
                order = int(self.Y,2) # i-th block to be scaled up, starts from 0
                unit_prob = self.errp / (2**n-1)
                remain = 2**n - 1 - order

                # nodes of lower bound and upper bound
                direction = 0 if tmp < p_lb else 1
                node_lb = self.tree.search_node(p_lb, direction)
                direction = 0 if tmp < p_ub else 1
                node_ub = self.tree.search_node(p_ub, direction)
                
                # step 1
                self.tree = node_lb.parent.rotate()
                self.tree.left.p = unit_prob * order
                self.tree.right.p = 1 - self.tree.left.p

                # step 2
                sub = node_ub.parent.rotate(subtree=True)
                if sub.parent is not None:
                    self.tree = sub.parent
                    sub = self.tree.right
                else:
                    self.tree = sub
                sub.left.p = sub.p * (1 - self.errp) / (1 - self.errp + unit_prob * remain)
                sub.right.p = 1 - sub.left.p  

            # split the new tree, figure out which block msg belongs to
            direction = 0 if self.tree.left.p < 1/2**n else 1
            j = 1
            node_ub = self.tree.search_node(1 / 2**n, 0)
            value_lb = 0
            value_ub = node_ub.start_value
            while value_lb <= msg and value_ub <= msg:
                value_lb = value_ub
                j += 1
                direction = 0 if self.tree.left.p < j/2**n else 1
                value_ub = self.tree.search_node(j/2**n, 0).start_value
            
            target = j - 1 # j-th interval
            print('value: {}\n'.format(value_lb))
            if abs(self.value - value_lb) < 1E-9 and i > 20:
                
                print("END: {}".format(i))
                return
            
            # TODO: which value should be picked up
            # self.value = (value_lb + value_ub) / 2 
            self.value = value_lb