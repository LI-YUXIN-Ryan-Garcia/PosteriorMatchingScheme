import os
import numpy as np
import matplotlib.pyplot as plt
from pms import PMS
from hamming import HammingCode
from utility import h, BSC_capacity

"""
A series test functions about standard Posterior Matching Scheme
*->  one-time PMS
*->  code length against channel use
*->  code length against transmission rate
*->  error probability against transmission rate

seq:    binary sequence
Px:     crossover probability
Pe:     error probability
h():    entropy function
"""

# parameter
# seq = '1001' # length 4
# seq = '110110110110111001000100011100110001' # length 36
seq = '11010111010101110101001010100010101101001010101110010101000100100101000100101001' *2# length 80
Px = 0.2
Pe = 0.001

def test_pms_once(sequence, Px, Pe):
    pms = PMS(Px, Pe)
    s, v, u = pms.transmit(sequence, max_channel_use=300)
    print("Result:\n- number of channel use:{}, len: {}\n- result value:{}".format(u,len(sequence),v))
    print("- binary sequence:{}\n- actual sequence:{}".format(s,sequence))
    if s == sequence:
        print("Correct!")
    else:
        print("Wrong!")
    print("Transmission Rate: {}".format(len(sequence) / u))
    print("Channel capacity:  {}".format(BSC_capacity(Px)))

# approximate runtime: 4min30s
def test_pms_len_against_channel_use(seq, Px):
    min_code_len = 2
    max_code_len = 54
    sample_size = 30
    Pe = 0.01
    channel_use = []
    for l in range(min_code_len,max_code_len+1):
        use = []
        for i in range(sample_size):
            pms = PMS(Px, Pe)
            s, v, u = pms.transmit(seq[:l], max_channel_use=500)
            if s == seq[:l]:
                use.append(u)
        channel_use.append(sum(use)/len(use))

    # plot
    x = np.array(range(min_code_len, max_code_len+1))
    y = channel_use
    plt.plot(x,y, label="channel use")
    plt.xlim([0, max_code_len+1])
    plt.xlabel("Binary sequence length")
    plt.ylabel("Number of channel use")
    plt.title("Pe={}, Px={}, size={}".format(Pe, Px, sample_size))  
    plt.legend(loc='upper left')
    plt.savefig(os.path.join("graph", "pms_len_channel_use.png"))
    plt.show()

#approximate runtime: 3h
def test_pms_len_against_tranx_rate(seq, Px):
    min_code_len = 1
    max_code_len = 60
    sample_size = 1000
    Pe = 0.01
    tranx_rate = []
    for l in range(min_code_len,max_code_len+1):
        rate = []
        for i in range(sample_size):
            pms = PMS(Px, Pe)
            s, v, u = pms.transmit(seq[:l], max_channel_use=500)
            if s == seq[:l]:
                rate.append(l/u)
        tranx_rate.append(sum(rate)/len(rate))
    capacity = BSC_capacity(Px)
    
    # plot
    x = np.array(range(min_code_len, max_code_len+1))
    y = tranx_rate
    plt.plot(x,y, label="tranx rate")
    plt.hlines(capacity, 0, max_code_len+1, colors="coral", label="capacity")
    plt.xlim([0, max_code_len+1])
    plt.xlabel("Binary sequence length")
    plt.ylabel("Transmission rate")
    plt.title("Pe={}, Px={}, size={}".format(Pe, Px, sample_size))  
    plt.legend(loc='lower right')
    plt.savefig(os.path.join("graph", "pms_len_tranx_rate.png"))
    plt.show()

def test_pms_err_prob_against_tranx_rate(sequence, Px):
    code_length = 36
    seq = sequence[:code_length]
    sample_size = 500
    error_probability = [ i/1000 for i in range(1,51)]
    tranx_rate = []
    for Pe in error_probability:
        rate = []
        for i in range(sample_size):
            pms = PMS(Px, Pe)
            s, v, u = pms.transmit(seq, max_channel_use=500)
            if s == seq:
                rate.append(code_length / u)
        tranx_rate.append(sum(rate)/len(rate))
    capacity = BSC_capacity(Px)

    # plot
    x = np.array(error_probability)
    y = tranx_rate
    x_max = max(error_probability)
    plt.plot(x,y, label="tranx rate")
    plt.hlines(capacity, 0, x_max, colors="coral", label="capacity")
    plt.xlim([0, x_max])
    plt.xlabel("Error Probability")
    plt.ylabel("Transmission rate")
    plt.title("Length={}, Px={}, size={}".format(code_length, Px, sample_size))  
    plt.legend(loc='lower right')
    plt.savefig(os.path.join("graph", "pms_Pe_tranx_rate.png"))
    plt.show()
    

# test_pms_once(seq[:3], Px, Pe)
# test_pms_len_against_channel_use(seq, Px)
test_pms_len_against_tranx_rate(seq, Px)
# test_pms_err_prob_against_tranx_rate(seq, Px)