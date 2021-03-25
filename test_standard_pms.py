import os
import numpy as np
import matplotlib.pyplot as plt
from pms import PMS
from utility import h, BSC_capacity, read_msg

"""
A series test functions about standard Posterior Matching Scheme
*->  one-time PMS
*->  message length against channel use (abandoned)
*->  message length against transmission rate
*->  error probability against transmission rate (abandoned)

msg:    message in binary
Px:     crossover probability
Pe:     error probability
h():    entropy function
"""

Px = 0.3
Pe = 0.01

def test_pms_once(msg_len, Px, Pe):
    # message
    msg = read_msg(msg_len)[233]

    # posterior matching
    pms = PMS(Px, Pe)
    s, v, u = pms.transmit(msg, max_channel_use=500) # str, value, use
    print("Result:\n- number of channel use: {}, len: {}\n- result value: {}".format(u,len(msg),v))
    print("- binary sequence:{}\n- actual sequence:{}".format(s,msg))
    if s == msg:
        print("Correct!")
        print("Transmission Rate: {}".format(len(msg) / u))
        print("Channel capacity:  {}".format(BSC_capacity(Px)))
    else:
        print("Wrong!")
    
# abandoned
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


def test_pms_len_against_tranx_rate(Px, Pe, cmt):
    """ Plot message length against transmission rate, write log file
    
    This function will record average transmission rate and channel use for
    each message length. Existent log file will be overwritten. 
    """

    min_code_len = 1
    max_code_len = 40
    sample_size = 700
    tranx_rate = np.zeros(max_code_len - min_code_len + 1)
    channel_use = np.zeros(max_code_len - min_code_len + 1)
    for l in range(min_code_len,max_code_len+1):
        rate = np.zeros(sample_size)
        use = np.zeros(sample_size)
        msg = read_msg(l)
        for i in range(sample_size):
            pms = PMS(Px, Pe)
            s, v, u = pms.transmit(msg[i], max_channel_use=500)
            if s == msg[i]:
                rate[i] = l / u
                use[i] = u
            print("Progress: {}%".format(np.round(100*((l-min_code_len)*sample_size+i) / (max_code_len+1-min_code_len) / sample_size,2)))
        # tranx_rate.append(sum(rate)/len(rate))
        tranx_rate[l-1] = np.mean(rate[np.nonzero(rate)])
        channel_use[l-1] = np.mean(use[np.nonzero(use)])
    capacity = BSC_capacity(Px)

    # log file
    fn_trx = "pms_len_tranx_rate_{}.txt".format(cmt)
    fn_trx = os.path.join('log', fn_trx)
    with open(fn_trx, 'w') as f:
        for rate in tranx_rate:
            f.write(str(rate)+'\n')

    fn_use = "pms_len_channel_use_{}.txt".format(cmt)
    fn_use = os.path.join('log', fn_use)
    with open(fn_use, 'w') as f:
        for use in channel_use:
            f.write(str(use)+'\n')

    # plot
    x = np.array(range(min_code_len, max_code_len+1))
    y = tranx_rate
    plt.plot(x,y, label="tranx rate")  
    plt.hlines(capacity, 0, max_code_len+1, colors="coral", label="capacity")

    plt.xlim([0, max_code_len+1])
    title = "Standard Posterior Matching Scheme"
    plt.title("{}: Pe={}, Px={}, size={}".format(title, Pe, Px, sample_size))
    plt.xlabel("Message length")
    plt.ylabel("Transmission rate")
      
    plt.legend(loc='lower right')
    plt.savefig(os.path.join("graph", "pms_len_tranx_rate_{}.png".format(cmt)))
    plt.show()

# abandoned
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
    
if __name__ == "__main__":
    comment = 'Px=0.3'
    # test_pms_once(100, Px, Pe)
    # test_pms_len_against_channel_use(seq, Px)
    test_pms_len_against_tranx_rate(Px, Pe, comment)
    # test_pms_err_prob_against_tranx_rate(seq, Px)