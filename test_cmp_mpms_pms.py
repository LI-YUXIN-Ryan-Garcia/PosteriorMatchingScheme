import os
import numpy as np
import matplotlib.pyplot as plt
from mpms import MPMS
from pms import PMS
from utility import h, BSC_capacity

# parameter
# seq = '1001' # length 4
seq = '110110110110111001000100011100110001'# length 36
# seq = '11010111010101110101001010100010101101001010101110010101000100100101000100101001' # length 80
Px = 0.2
Pe = 0.001

def test_cmp_code_len_against_tranx_rate(sequence, Px, Pe):
    min_code_len = 1
    max_code_len = len(sequence)
    sample_size = 300
    capacity = BSC_capacity(Px)
    tranx_rate_pms = []
    tranx_rate_mpms = []
    est_mpms_cap = [] # estimated MPMS capacity
    for l in range(min_code_len, max_code_len+1):
        # modified posterior matching scheme
        rate_mpms = []
        hamming_len = 0
        for i in range(sample_size):
            mpms = MPMS(Px, Pe)
            s, u, L = mpms.transmit(sequence[:l], max_channel_use=500, err_num=l)
            hamming_len = L
            if s == sequence[:l]:
                rate_mpms.append(l/u/L)
        tranx_rate_mpms.append(sum(rate_mpms)/len(rate_mpms))
        est_mpms_cap.append(capacity * l / hamming_len)
        
        # standard posterior matching scheme
        rate_pms = []
        for j in range(sample_size):
            pms = PMS(Px, Pe)
            s, v, u = pms.transmit(sequence[:l], max_channel_use=500)
            if s == seq[:l]:
                rate_pms.append(l/u)
        tranx_rate_pms.append(sum(rate_pms)/len(rate_pms))

    # plot
    x = np.array(range(min_code_len, max_code_len+1))
    y1 = tranx_rate_mpms
    y2 = tranx_rate_pms
    z = est_mpms_cap
    plt.plot(x, y1, label="MPMS tranx rate")
    plt.plot(x, y2, color='coral', label="PMS tranx rate")
    plt.plot(x, z, color='lime', label="estimated capacity")
    plt.hlines(capacity, 0, max_code_len+1, colors='red', label="capacity")
    plt.xlim([0, max_code_len+1])
    plt.ylim([0, capacity * 1.1])
    plt.xlabel("Binary sequence length")
    plt.ylabel("Transmission rate")
    plt.title("Pe={}, Px={}, size={}".format(Pe, Px, sample_size))  
    plt.legend(loc='lower right')
    plt.savefig(os.path.join("graph", "cmp_len_tranx_rate.png"))
    plt.show()

test_cmp_code_len_against_tranx_rate(seq[:25], Px, Pe)