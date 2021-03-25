import os
import numpy as np
from scipy.special import comb

# entropy function
def h(x): return x * np.log2(1/x) + (1-x) * np.log2(1/(1-x))

# capacity
def BSC_capacity(alpha): return 1 - h(alpha)

def BSC_Hamming_capacity(n, k, rho):
    """
    n:   hamming block length
    k:   hamming message length
    rho: error probability, P(u_k != v_k)
    C = I(u_k, v_k) / n = (H(v_k) - H(v_k|u_k)) / n 
    """
    H_vk_given_uk = (1-rho) * np.log2(1/(1-rho)) + rho * np.log2((2**k-1) / rho)
    return (k - H_vk_given_uk) / n

# hamming code error probability
def hamming_err_prob(Px, msg_len, block_len):
    d = 3
    sum_p = 0
    for i in range(d//2 + 1):
        sum_p += comb(block_len, i, exact=True) * Px**i * (1-Px)**(block_len-i)
    return 1 - sum_p

def hamming_LOEP(Px, block_len):
    """ Error probability of leading order"""

    return 3 * comb(block_len, 2) * Px * Px / block_len

def read_msg(L):
    """ Read message file and return message"""
    fn = 'msg_{}.txt'.format(L)
    fn = os.path.join('message', fn)
    with open(fn, 'r') as f:
        msg_list = f.readlines()
        msg_list = [s.strip() for s in msg_list]
    
    return msg_list