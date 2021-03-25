import os
import numpy as np
import matplotlib.pyplot as plt
from hamming import HammingCode
from utility import h, BSC_capacity, BSC_Hamming_capacity, hamming_err_prob, hamming_LOEP

Px = 0.2
Pe = 0.01

def test_cmp_code_len_against_tranx_rate(Px, Pe, msgL=4, est=False):
    """ Read logs, plot message length against transmission rate and capacity
    
    This function will read log files recording average transmission rate and 
    channel use for each message length and compare them with channel complexity. 
    """

    min_msg_len = 1
    max_msg_len = 80
    sample_size = 1000
    
    # hamming code
    hmsg_len =  msgL
    hblk_len = HammingCode.calc_redundant_bits(hmsg_len) + hmsg_len
    hrate = "{}/{}".format(hmsg_len,hblk_len)
    name = 'Hamming({},{})'.format(hblk_len,hmsg_len)

    # read log files
    fn_pms = 'pms_len_tranx_rate.txt'
    fn_HEP = 'mpms_len_tranx_rate_({},{})_Ps=HEP.txt'.format(hblk_len,hmsg_len)
    fn_pms = os.path.join('log', fn_pms)
    fn_HEP = os.path.join('log', fn_HEP)
    
    with open(fn_pms, 'r') as f:
        rate_list_pms = f.readlines()
        rate_pms = np.array([np.float32(r.strip()) for r in rate_list_pms])
    
    with open(fn_HEP, 'r') as f:
        rate_list_HEP = f.readlines()
        rate_HEP = np.array([np.float32(r.strip()) for r in rate_list_HEP])
    
    if est: # estimate scaled probability
        fn_Px = 'mpms_len_tranx_rate_({},{})_Ps=Px.txt'.format(hblk_len,hmsg_len)
        fn_LOEP = 'mpms_len_tranx_rate_({},{})_Ps=LOEP.txt'.format(hblk_len,hmsg_len)
        fn_Px = os.path.join('log', fn_Px)
        fn_LOEP = os.path.join('log', fn_LOEP)
        
        with open(fn_Px, 'r') as f:
            rate_list_Px = f.readlines()
            rate_Px = np.array([np.float32(r.strip()) for r in rate_list_Px])
        
        with open(fn_LOEP, 'r') as f:
            rate_list_LOEP = f.readlines()
            rate_LOEP = np.array([np.float32(r.strip()) for r in rate_list_LOEP])

    # capacity
    capacity = BSC_capacity(Px)
    rho1 = hamming_err_prob(Px, hmsg_len, hblk_len)
    capacity_with_HEP = BSC_Hamming_capacity(hblk_len, hmsg_len, rho1)
    if est:
        rho2 = hamming_LOEP(Px, hblk_len)
        capacity_with_LOEP = BSC_Hamming_capacity(hblk_len, hmsg_len, rho2)

    # data to be plotted
    x = np.array(range(min_msg_len, max_msg_len+1))
    y1 = rate_pms
    y2 = rate_HEP
    y3 = rate_pms * hmsg_len / hblk_len
    c1 = capacity
    c2 = capacity_with_HEP
    c3 = capacity * hmsg_len / hblk_len
    if est:
        y4 = rate_Px
        y5 = rate_LOEP
        c5 = capacity_with_LOEP

    # plot
    figname = "cmp_len_tranx_rate_{}.png".format(name)
    plt.plot(x, y1, label="PMS", color='tab:orange')
    plt.plot(x, y2, label=name, color='tab:blue')
    plt.plot(x, y3, label="PMS x {}".format(hrate), color='tab:green')
    plt.hlines(c1, 0, max_msg_len+1, colors='red', label="capacity")
    plt.hlines(c2, 0, max_msg_len+1, colors='m', label="cap of HEP")
    if est:
        plt.plot(x, y4, label="Crossover", color='saddlebrown')
        plt.plot(x, y5, label="LOEP", color='black')
        # plt.hlines(c5, 0, max_msg_len+1, colors='cyan', label="cap of LOEP")
        figname = "cmp_len_tranx_rate_{}_EST.png".format(name)

    plt.xlim([0, max_msg_len+1])
    plt.ylim([0, c1 * 1.1])
    plt.xlabel("Message length")
    plt.ylabel("Transmission rate")
    plt.legend(loc='lower right')
    plt.title("Comparison: Pe={},Px={},size={},{}".format(Pe,Px,sample_size,name))
    plt.savefig(os.path.join("graph", figname))
    plt.show()


def test_cmp_hamming_7_4_diff_Px(Pe):
    min_msg_len = 1
    max_msg_len = 34
    sample_size = 700

    # read log files
    fn_pms_1 = 'pms_len_tranx_rate_Px=0.1.txt'
    fn_pms_2 = 'pms_len_tranx_rate.txt'
    fn_pms_3 = 'pms_len_tranx_rate_Px=0.3.txt'
    fn_HEP_1 = 'mpms_len_tranx_rate_(7,4)_Px=0.1.txt'
    fn_HEP_2 = 'mpms_len_tranx_rate_(7,4)_Ps=HEP.txt'
    fn_HEP_3 = 'mpms_len_tranx_rate_(7,4)_Px=0.3.txt'
    fn_pms_1 = os.path.join('log', fn_pms_1)
    fn_pms_2 = os.path.join('log', fn_pms_2)
    fn_pms_3 = os.path.join('log', fn_pms_3)
    fn_HEP_1 = os.path.join('log', fn_HEP_1)
    fn_HEP_2 = os.path.join('log', fn_HEP_2)
    fn_HEP_3 = os.path.join('log', fn_HEP_3)
    
    with open(fn_pms_1, 'r') as f:
        rate_list_pms_1 = f.readlines()
        rate_pms_1 = np.array([np.float32(r.strip()) for r in rate_list_pms_1])
    
    with open(fn_pms_2, 'r') as f:
        rate_list_pms_2 = f.readlines()
        rate_pms_2 = np.array([np.float32(r.strip()) for r in rate_list_pms_2])

    with open(fn_pms_3, 'r') as f:
        rate_list_pms_3 = f.readlines()
        rate_pms_3 = np.array([np.float32(r.strip()) for r in rate_list_pms_3])
    
    with open(fn_HEP_1, 'r') as f:
        rate_list_HEP_1 = f.readlines()
        rate_HEP_1 = np.array([np.float32(r.strip()) for r in rate_list_HEP_1])

    with open(fn_HEP_2, 'r') as f:
        rate_list_HEP_2 = f.readlines()
        rate_HEP_2 = np.array([np.float32(r.strip()) for r in rate_list_HEP_2])

    with open(fn_HEP_3, 'r') as f:
        rate_list_HEP_3 = f.readlines()
        rate_HEP_3 = np.array([np.float32(r.strip()) for r in rate_list_HEP_3])

    # capacity
    capacity_1 = BSC_capacity(0.1)
    capacity_2 = BSC_capacity(0.2)
    capacity_3 = BSC_capacity(0.3)
    rho1 = hamming_err_prob(0.1, 4, 7)
    rho2 = hamming_err_prob(0.2, 4, 7)
    rho3 = hamming_err_prob(0.3, 4, 7)
    capacity_with_HEP_1 = BSC_Hamming_capacity(7, 4, rho1)
    capacity_with_HEP_2 = BSC_Hamming_capacity(7, 4, rho2)
    capacity_with_HEP_3 = BSC_Hamming_capacity(7, 4, rho3)

    # data to be plotted
    x = np.array(range(min_msg_len, max_msg_len+1))
    y1 = rate_pms_1[min_msg_len:max_msg_len+1]
    y2 = rate_pms_2[min_msg_len:max_msg_len+1]
    y3 = rate_pms_3[min_msg_len:max_msg_len+1]
    y4 = rate_HEP_1[min_msg_len:max_msg_len+1]
    y5 = rate_HEP_2[min_msg_len:max_msg_len+1]
    y6 = rate_HEP_3[min_msg_len:max_msg_len+1]
    c1 = capacity_1
    c2 = capacity_2
    c3 = capacity_3
    c4 = capacity_with_HEP_1
    c5 = capacity_with_HEP_2
    c6 = capacity_with_HEP_3

    # plot
    figname = "cmp_len_tranx_rate_diff_Px.png"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(x, y1, label="PMS .1", color='tab:blue')
    ax1.plot(x, y4, label="MPMS .1", color='tab:orange')
    ax1.hlines(c1, 0, max_msg_len+1, colors='red', label="cap PMS .1")
    ax1.set_xlim([0, max_msg_len+1])
    ax1.set_ylim([0, c1 * 1.1])
    ax1.set_xlabel("Message length")
    ax1.legend(loc='lower right')
    ax1.spines['bottom'].set_linewidth(0.5)

    ax2.plot(x, y2, label="PMS .2", color='tab:blue')
    ax2.plot(x, y5, label="MPMS .2", color='tab:orange')
    ax2.hlines(c2, 0, max_msg_len+1, colors='red', label="cap PMS .2")
    ax2.set_xlim([0, max_msg_len+1])
    ax2.set_ylim([0, c1 * 1.1])
    ax2.set_xlabel("Message length")
    ax2.legend(loc='upper right')

    ax3.plot(x, y3, label="PMS .3", color='tab:blue')
    ax3.plot(x, y6, label="MPMS .3", color='tab:orange')
    ax3.hlines(c3, 0, max_msg_len+1, colors='red', label="cap PMS .3")
    ax3.set_xlim([0, max_msg_len+1])
    ax3.set_ylim([0, c1 * 1.1])
    ax3.set_xlabel("Message length")
    ax3.legend(loc='upper right')

    # plt.hlines(c4, 0, max_msg_len+1, colors='m', label="cap MPMS .1")
    # plt.hlines(c5, 0, max_msg_len+1, colors='m', label="cap MPMS .2")
    # plt.hlines(c6, 0, max_msg_len+1, colors='m', label="cap MPMS .3")
    
    fig.suptitle("Comparison: Pe={}, size={}, diff Px".format(Pe,Px,sample_size))
    plt.savefig(os.path.join("graph", figname))
    plt.show()

def test_mismatched_Px(Px, Pe, msgL=4):

    min_msg_len = 1
    max_msg_len = 39
    sample_size = "700/1000" 
    
    # hamming code
    hmsg_len =  msgL
    hblk_len = HammingCode.calc_redundant_bits(hmsg_len) + hmsg_len
    hrate = "{}/{}".format(hmsg_len,hblk_len)
    name = 'Hamming({},{})'.format(hblk_len,hmsg_len)

    # read log files
    fn_pms = 'pms_len_tranx_rate.txt'
    fn_HEP = 'mpms_len_tranx_rate_({},{})_Ps=HEP.txt'.format(hblk_len,hmsg_len)
    fn_mm = 'mpms_len_tranx_rate_({},{})_mismatched_Px.txt'.format(hblk_len,hmsg_len)
    fn_pms = os.path.join('log', fn_pms)
    fn_HEP = os.path.join('log', fn_HEP)
    fn_mm = os.path.join('log', fn_mm)
    
    with open(fn_pms, 'r') as f:
        rate_list_pms = f.readlines()
        rate_pms = np.array([np.float32(r.strip()) for r in rate_list_pms])
    
    with open(fn_HEP, 'r') as f:
        rate_list_HEP = f.readlines()
        rate_HEP = np.array([np.float32(r.strip()) for r in rate_list_HEP])

    with open(fn_mm, 'r') as f:
        rate_list_mm = f.readlines()
        rate_mm = np.array([np.float32(r.strip()) for r in rate_list_mm])

    # capacity
    capacity = BSC_capacity(Px)
    rho1 = hamming_err_prob(Px, hmsg_len, hblk_len)
    capacity_with_HEP = BSC_Hamming_capacity(hblk_len, hmsg_len, rho1)

    # data to be plotted
    x = np.array(range(min_msg_len, max_msg_len+1))
    y1 = rate_pms[min_msg_len:max_msg_len+1]
    y2 = rate_HEP[min_msg_len:max_msg_len+1]
    y3 = rate_mm[min_msg_len:max_msg_len+1]
    c1 = capacity
    c2 = capacity_with_HEP

    # plot
    figname = "cmp_len_tranx_rate_{}_with_mismatched_Px.png".format(name)
    plt.plot(x, y1, label="PMS", color='tab:orange')
    plt.plot(x, y2, label=name, color='tab:blue')
    plt.plot(x, y3, label="mismatched", color='tab:green')
    plt.hlines(c1, 0, max_msg_len+1, colors='red', label="capacity")
    plt.hlines(c2, 0, max_msg_len+1, colors='m', label="cap of HEP")

    plt.xlim([0, max_msg_len+1])
    plt.ylim([0, c1 * 1.1])
    plt.xlabel("Message length")
    plt.ylabel("Transmission rate")
    plt.legend(loc='lower right')
    plt.title("Comparison: Pe={},Px={},size={},{}".format(Pe,Px,sample_size,name))
    plt.savefig(os.path.join("graph", figname))
    plt.show()
    

if __name__ == '__main__':
    estimate = True
    # test_cmp_code_len_against_tranx_rate(Px, Pe, msgL=4, est=estimate)
    # test_cmp_hamming_7_4_diff_Px(Pe)
    test_mismatched_Px(Px, Pe, msgL=4)
    
    