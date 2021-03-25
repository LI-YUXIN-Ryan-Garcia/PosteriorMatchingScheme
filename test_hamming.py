import numpy as np
from hamming import HammingCode
from utility import hamming_err_prob, BSC_Hamming_capacity
from scipy.special import comb

# msg_len = 4
# block_len = HammingCode.calc_redundant_bits(msg_len) + msg_len
# msg_bin = [format(i, str(msg_len)+'b').replace(' ','0') for i in range(2**msg_len)]
# # print(msg_bin)
# code = {}
# for i in range(2**msg_len):
#     m = msg_bin[i]
#     h = HammingCode(m)
#     code[m] = h.encode()
# print(code)

# d = 3
# sum_p = 0
# Px = 0.2
# for i in range(d//2 + 1):
#     print(comb(block_len,i))
#     sum_p += comb(block_len, i, exact=True) * Px**i * (1-Px)**(block_len-i)

# hamming_err_p = 1 - sum_p
# print(hamming_err_p)
# test_gen_hamming_code()

def test_diff_hamming_err_prob(msg_Lmin, msg_Lmax, Px, display=False):
    msg_len = np.arange(msg_Lmin, msg_Lmax+1)
    blk_len = np.arange(msg_Lmin, msg_Lmax+1)
    p = np.zeros((msg_Lmax - msg_Lmin + 1))
    for ml in msg_len:
        r = HammingCode.calc_redundant_bits(ml)
        bl = ml + r
        blk_len[ml - msg_Lmin] = bl
        p[ml - msg_Lmin] = hamming_err_prob(Px, ml, bl)
        if display:
            print('hamming({},{}): {}'.format(bl,ml,p[ml-msg_Lmin]))
    return msg_len, blk_len, p

def test_diff_capacity_with_pms(msg_Lmin, msg_Lmax, Px, display=False):
    msg_len, blk_len, prob = test_diff_hamming_err_prob(msg_Lmin, msg_Lmax, Px)
    capcity = np.zeros(msg_Lmax - msg_Lmin + 1)
    for i in range(len(msg_len)):
        ml, bl, rho = msg_len[i], blk_len[i], prob[i]
        cap = BSC_Hamming_capacity(bl, ml, rho)
        capcity[i] = cap
        if display:
            print('Capacity of Hamming({},{}): {}'.format(bl,ml,cap))
    return capcity

if __name__ == '__main__':
    Px = 0.2
    m1, m2 = 1, 15
    # test_diff_hamming_err_prob(m1,m2,Px,display=True)
    test_diff_capacity_with_pms(m1,m2,Px,display=True)