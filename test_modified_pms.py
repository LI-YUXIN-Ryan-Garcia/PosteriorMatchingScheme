import os
import numpy as np
import matplotlib.pyplot as plt
from mpms import MPMS
from utility import h, BSC_capacity, read_msg, hamming_err_prob, BSC_Hamming_capacity
from hamming import HammingCode

"""
A series test functions about modified Posterior Matching Scheme
*->  one-time MPMS, all error(s) can be corrected
*->  one-time MPMS, some error(s) can't be corrected due to limitation of the 
    ability of hamming code
*->  number of error(s) in linear code against correct transmission ratio, some
    error(s) may not be corrected.
*->  code length against transmission rate 

msg:    message in binary
Px:     crossover probability
Pe:     error probability
h():    entropy function
"""

Px = 0.2
Pe = 0.01

# abandoned
def test_mpms_once_with_errors_all_corrected(sequence, Px, Pe):
    mpms = MPMS(Px, Pe)
    hamming_msg_len = 4
    s, u, l = mpms.transmit(sequence, max_channel_use=300, err_num=1, msg_len=hamming_msg_len)
    print("Result:\n- number of channel use:{}, len: {} {}".format(u,len(sequence),l))
    print("- binary sequence:{}\n- actual sequence:{}".format(s,sequence))
    if s == sequence:
        print("Correct!")
    else:
        print("Wrong!")
    print("Transmission Rate: {}".format(len(sequence) / u / l))
    print("Channel x linear:  {}".format(BSC_capacity(Px) * hamming_msg_len / l))
    print("Channel capacity:  {}".format(BSC_capacity(Px)))

def test_mpms_once_with_errors_not_all_corrected(msg_len, Px, Pe):
    # message and hamming code
    msg = read_msg(msg_len)[233]
    hmsg_len = 10 # hamming message length
    hblk_len = HammingCode.calc_redundant_bits(hmsg_len) + hmsg_len
    print('Hamming({},{})'.format(hblk_len, hmsg_len))

    # modified posterior matching
    mpms = MPMS(Px, Pe)
    s, u, l = mpms.transmit(msg, max_channel_use=500, err_num=None, msg_len=hmsg_len)
    print("Result:\n- number of channel use:{}, len: {} {}".format(u,len(msg),l))
    print("- binary sequence:{}\n- actual sequence:{}".format(s,msg))
    if s == msg:
        print("Correct!")
        print("Transmission Rate: {}".format(len(msg) / u / hblk_len))
        print("Channel capacity:  {}".format(BSC_capacity(Px)))
        print("Channel x linear:  {}".format(BSC_capacity(Px) * hmsg_len / hblk_len))
    else:
        print("Wrong!")
    
# abandoned
def test_mpms_err_num_against_err_prob_with_not_errors_all_corrected(sequence, Px, Pe):
    hamming_msg_len = 20
    min_err_num = 1
    max_err_num = hamming_msg_len
    sample_size = 300
    succ_tranx = []
    for n in range(min_err_num, max_err_num+1):
        count = 0
        for i in range(sample_size):
            mpms = MPMS(Px, Pe)
            s, u, l = mpms.transmit(sequence, max_channel_use=500, err_num=n, msg_len=hamming_msg_len)
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

def test_mpms_len_against_tranx_rate_with_not_errors_all_corrected(Px, Pe, cmt:str):
    """ Plot message length against transmission rate, write log file
    
    This function will record average transmission rate and channel use for
    each message length. Existent log file will be overwritten. 
    """

    # hamming code
    hmsg_len = 4
    hblk_len = hblk_len = HammingCode.calc_redundant_bits(hmsg_len) + hmsg_len
    
    # modified posterior mathcing
    min_msg_len = 1
    max_msg_len = 40
    sample_size = 700
    tranx_rate = np.zeros(max_msg_len - min_msg_len + 1)
    channel_use = np.zeros(max_msg_len - min_msg_len + 1)
    for l in range(min_msg_len, max_msg_len+1):
        rate = np.zeros(sample_size)
        use = np.zeros(sample_size)
        msg = read_msg(l)
        for i in range(sample_size):
            mpms = MPMS(Px, Pe)
            s, u, L = mpms.transmit(msg[i], max_channel_use=500, err_num=None, msg_len=hmsg_len)
            if s == msg[i]:
                rate[i] = l / u / hblk_len
                use[i] = u
            print("Progress: {}%".format(np.round(100*((l-min_msg_len)*sample_size+i) / (max_msg_len+1-min_msg_len) / sample_size,2)))
        # tranx_rate.append(sum(rate)/len(rate))
        tranx_rate[l-1] = np.mean(rate[np.nonzero(rate)])
        channel_use[l-1] = np.mean(use[np.nonzero(use)])

    # log file
    fn_trx = "mpms_len_tranx_rate_({},{})_{}.txt".format(hblk_len, hmsg_len, cmt)
    fn_trx = os.path.join('log', fn_trx)
    with open(fn_trx, 'w') as f:
        for rate in tranx_rate:
            f.write(str(rate)+'\n')

    fn_use = "mpms_len_channel_use_({},{})_{}.txt".format(hblk_len, hmsg_len, cmt)
    fn_use = os.path.join('log', fn_use)
    with open(fn_use, 'w') as f:
        for use in channel_use:
            f.write(str(use)+'\n')

    # plot
    x = np.array(range(min_msg_len, max_msg_len+1))
    y1 = tranx_rate
    c1 = BSC_capacity(Px)
    c2 = BSC_Hamming_capacity(hblk_len,hmsg_len,hamming_err_prob(Px,hmsg_len,hblk_len))
    plt.plot(x, y1, label="tranx rate")
    plt.hlines(c1, 0, max_msg_len+1, colors="coral", label="BSC capacity")
    plt.hlines(c2, 0, max_msg_len+1, color='lime', label="Cap with Hamming")
    
    plt.xlim([0, max_msg_len+1])
    title = 'Modified Posterior Matching Scheme: Pe={}, Px={}, '.format(Pe, Px)
    plt.title("{}size={}, hamming({},{})".format(title, sample_size, hblk_len, hmsg_len))
    plt.xlabel("Message length")
    plt.ylabel("Transmission rate")
      
    plt.legend(loc='lower right')
    figname = "mpms_len_tranx_rate_({},{})_{}.png".format(hblk_len, hmsg_len, cmt)
    plt.savefig(os.path.join("graph", figname))
    plt.show()

def test_cmp_diff_hamming(Px, Pe, cmt):
    min_msg_len = 1
    max_msg_len = 80
    sample_size = 1000

    # read log files
    fn_pms = 'pms_len_tranx_rate.txt'
    fn_52 = 'mpms_len_tranx_rate_(5,2)_{}.txt'.format(cmt)
    fn_63 = 'mpms_len_tranx_rate_(6,3)_{}.txt'.format(cmt)
    fn_74 = 'mpms_len_tranx_rate_(7,4)_{}.txt'.format(cmt)
    fn_1410 = 'mpms_len_tranx_rate_(14,10)_{}.txt'.format(cmt)

    fn_pms = os.path.join('log', fn_pms)
    fn_52 = os.path.join('log', fn_52)
    fn_63 = os.path.join('log', fn_63)
    fn_74 = os.path.join('log', fn_74)
    fn_1410 = os.path.join('log', fn_1410) 
    with open(fn_pms, 'r') as f:
        rate_list_pms = f.readlines()
        rate_pms = np.array([np.float32(r.strip()) for r in rate_list_pms])

    with open(fn_52, 'r') as f:
        rate_list_52 = f.readlines()
        rate_52 = np.array([np.float32(r.strip()) for r in rate_list_52])
    
    with open(fn_63, 'r') as f:
        rate_list_63 = f.readlines()
        rate_63 = np.array([np.float32(r.strip()) for r in rate_list_63])
    
    with open(fn_74, 'r') as f:
        rate_list_74 = f.readlines()
        rate_74 = np.array([np.float32(r.strip()) for r in rate_list_74])

    with open(fn_1410, 'r') as f:
        rate_list_1410 = f.readlines()
        rate_1410 = np.array([np.float32(r.strip()) for r in rate_list_1410])

    # capacity
    rho1 = hamming_err_prob(Px, 2, 5)
    rho2 = hamming_err_prob(Px, 3, 6)
    rho3 = hamming_err_prob(Px, 4, 7)
    rho4 = hamming_err_prob(Px, 10, 14)
    capacity = BSC_capacity(Px)
    capacity_52 = BSC_Hamming_capacity(5, 2, rho1)
    capacity_63 = BSC_Hamming_capacity(6, 3, rho2)
    capacity_74 = BSC_Hamming_capacity(7, 4, rho3)
    capacity_1410 = BSC_Hamming_capacity(14, 10, rho4)

    # plot
    x = np.array(range(min_msg_len, max_msg_len+1))
    y1 = rate_pms
    y2 = rate_52
    y3 = rate_63
    y4 = rate_74
    y5 = rate_1410
    c1 = capacity
    c2 = capacity_52
    c3 = capacity_63
    c4 = capacity_74
    c5 = capacity_1410

    plt.plot(x, y1, label="PMS", color='tab:orange')
    plt.plot(x, y2, label="(5,2)", color='black')
    plt.plot(x, y3, label="(6,3)", color='saddlebrown')
    plt.plot(x, y4, label="(7,4)", color='tab:blue')
    plt.plot(x, y5, label="(14,10)", color='tab:green')
    plt.hlines(c1, 0, max_msg_len+1, colors='red', label="capacity")
    plt.hlines(c2, 0, max_msg_len+1, colors='gold', label="cap (5,2)")
    plt.hlines(c3, 0, max_msg_len+1, colors='coral', label="cap (6,3)")
    plt.hlines(c4, 0, max_msg_len+1, colors='grey', label="cap (7,4)")
    plt.hlines(c5, 0, max_msg_len+1, colors='blue', label="cap (14,10)")

    plt.xlim([0, max_msg_len+1])
    plt.ylim([0, c1 * 1.1])
    plt.xlabel("Message length")
    plt.ylabel("Transmission rate")
    plt.title("Comparison: Pe={}, Px={}, size={}, {}".format(Pe,Px,sample_size,cmt))  
    plt.legend(loc='lower right')
    figname = "cmp_hamming_{}.png".format(cmt)
    plt.savefig(os.path.join("graph", figname))
    plt.show()

if __name__ == "__main__":
    # test_mpms_once_with_errors_all_corrected(seq, Px, Pe)
    # test_mpms_once_with_errors_not_all_corrected(68, Px, Pe)
    # test_mpms_err_num_against_err_prob_with_not_errors_all_corrected(seq, Px, Pe)
    # comment = 'Px=0.1'
    comment = 'mismatched_Px'
    test_mpms_len_against_tranx_rate_with_not_errors_all_corrected(Px, Pe, comment)
    comment = "Ps=HEP"
    # test_cmp_diff_hamming(Px, Pe, comment)
