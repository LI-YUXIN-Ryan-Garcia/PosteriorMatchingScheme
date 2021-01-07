import os
import numpy as np
import matplotlib.pyplot as plt
from mpms import MPMS
from utility import h, BSC_capacity

"""
A series test functions about modified Posterior Matching Scheme
*->  one-time MPMS, all error(s) can be corrected
*->  one-time MPMS, some error(s) can't be corrected due to limitation of the 
    ability of hamming code
*->  number of error(s) in linear code against correct transmission ratio, some
    error(s) may not be corrected.
*->  code length against transmission rate 

seq:    binary sequence
Px:     crossover probability
Pe:     error probability
h():    entropy function
"""

# parameter
# seq = '1001' # length 4
seq = '110110110110111001000100011100110001'# length 36
# seq = '11010111010101110101001010100010101101001010101110010101000100100101000100101001' # length 80
Px = 0.2
Pe = 0.001

def test_mpms_once_with_errors_all_corrected(sequence, Px, Pe):
    mpms = MPMS(Px, Pe)
    s, u, l = mpms.transmit(sequence, max_channel_use=300, err_num=1)
    print("Result:\n- number of channel use:{}, len: {} {}".format(u,len(sequence),l))
    print("- binary sequence:{}\n- actual sequence:{}".format(s,sequence))
    if s == sequence:
        print("Correct!")
    else:
        print("Wrong!")
    print("Transmission Rate: {}".format(len(sequence) / u / l))
    print("Channel x linear:  {}".format(BSC_capacity(Px) * len(sequence) / l))
    print("Channel capacity:  {}".format(BSC_capacity(Px)))

def test_mpms_once_with_errors_not_all_corrected(sequence, Px, Pe):
    mpms = MPMS(Px, Pe)
    s, u, l = mpms.transmit(sequence, max_channel_use=300, err_num=10)
    print("Result:\n- number of channel use:{}, len: {} {}".format(u,len(sequence),l))
    print("- binary sequence:{}\n- actual sequence:{}".format(s,sequence))
    if s == sequence:
        print("Correct!")
    else:
        print("Wrong!")
    print("Transmission Rate: {}".format(len(sequence) / u / l))
    print("Channel x linear:  {}".format(BSC_capacity(Px) * len(sequence) / l))
    print("Channel capacity:  {}".format(BSC_capacity(Px)))

def test_mpms_err_num_against_err_prob_with_not_errors_all_corrected(sequence, Px, Pe):
    min_err_num = 1
    max_err_num = len(sequence)
    sample_size = 300
    succ_tranx = []
    for n in range(min_err_num, max_err_num+1):
        count = 0
        for i in range(sample_size):
            mpms = MPMS(Px, Pe)
            s, u, l = mpms.transmit(sequence, max_channel_use=500, err_num=n)
            if s == sequence:
                count += 1
        succ_tranx.append(count/sample_size)
     
    #plot
    x = np.array(range(min_err_num, max_err_num+1))
    y = succ_tranx
    plt.plot(x, y, label='correct tranx ratio')
    plt.xlabel("The number of error(s)")
    plt.ylabel("Correct tranx ratio")
    plt.title("Pe={}, Px={}, size={}, len={}".format(Pe, Px, sample_size, len(seq))) 
    plt.legend(loc='upper right')
    plt.savefig(os.path.join("graph", "mpms_err_num_err_prob.png"))
    plt.show()

def test_mpms_len_against_tranx_rate_with_not_errors_all_corrected(sequence, Px, Pe):
    min_code_len = 2
    max_code_len = len(sequence)
    sample_size = 300
    capacity = BSC_capacity(Px)
    tranx_rate = []
    tmp = []
    for l in range(min_code_len, max_code_len+1):
        rate = []
        hamming_len = 0
        for i in range(sample_size):
            mpms = MPMS(Px, Pe)
            s, u, L = mpms.transmit(sequence[:l], max_channel_use=500, err_num=l)
            hamming_len = L
            if s == sequence[:l]:
                rate.append(l/u/L)
        tranx_rate.append(sum(rate)/len(rate))
        tmp.append(capacity * l / hamming_len)

    # plot
    x = np.array(range(min_code_len, max_code_len+1))
    y = tranx_rate
    z = tmp
    plt.plot(x,y, label="tranx rate")
    plt.plot(x,z, label='approximate capacity')
    plt.hlines(capacity, 0, max_code_len+1, colors="coral", label="capacity")
    plt.xlim([0, max_code_len+1])
    plt.xlabel("Binary sequence length")
    plt.ylabel("Transmission rate")
    plt.title("Pe={}, Px={}, size={}".format(Pe, Px, sample_size))  
    plt.legend(loc='lower right')
    plt.savefig(os.path.join("graph", "mpms_len_tranx_rate.png"))
    plt.show()


# test_mpms_once_with_errors_all_corrected(seq, Px, Pe)
# test_mpms_once_with_errors_not_all_corrected(seq, Px, Pe)
# test_mpms_err_num_against_err_prob_with_not_errors_all_corrected(seq, Px, Pe)
test_mpms_len_against_tranx_rate_with_not_errors_all_corrected(seq[:25], Px, Pe)
