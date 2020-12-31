from mpms import MPMS
from pms import PMS
from hamming import HammingCode
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import statistics as stat

# entropy function
def h(x): return x * np.log2(1/x) + (1-x) * np.log2(1/(1-x))
# capacity
def BSC_capacity(alpha): return 1 - h(alpha)

# message: 0.6 0.0314159 0.73248

'''
Available Testing
'''
test = 2
# np.random.seed(0)
# hamming code testing:
if test == 1: 
    # hamming = HammingCode('1001')
    # print(hamming.calcRedundantBits(1))
    # print(hamming. genCode())
    # hamming.detectError('1001100', nr=2)
    # print(hamming.decode('1001100'[::-1]))
    
    msg = '0000111'
    code = '11011011011'
    hamming = HammingCode(msg)
    r = hamming.calcRedundantBits()
    x = hamming. genCode()
    n = hamming.detectError(code, nr=r)
    y = hamming.decode('11011011011'[::-1])
    print("msg: {}, with redundant bit: {}\n{}\n{}".format(msg, r, x, y))
    print('Error at {}'.format(n))
    
# standard posterior matching scheme testing
elif test == 2: 
    #sequence = '11010111010101110101001010100010101101001010101110010101000100100101000100101001'
    # sequence = '110110110110111001000100011100110001'
    sequence = '1001'
    crossover_prob = 0.2
    size = 1
    all_rounds = []
    correct_rounds = []
    for i in range(size):
        pms = PMS(crossover_prob)
        s, v, r = pms.transmit(sequence)
        all_rounds.append(r)
        print("Result:\n-rounds:{}\n-real number:{}\n-binary sequence:{}".format(r,v,s))
        print("-actual sequence:{} len: {}".format(sequence, len(sequence)))
        diff = bin(int(s,2) ^ int(sequence,2))[2:]
        num_e = 0
        for c in diff:
            if c == '1':
                num_e += 1
        print("error rate: {}".format(num_e / len(sequence)))
        if s == sequence:
            print("Correct!")
            correct_rounds.append(r)
        else:
            print("Wrong!")
        print("Transmission Rate: {}".format(len(sequence)/ r))
    
    r1, r2 = stat.median(all_rounds), None
    print("Average rounds: total: {} correct: {}".format(r1, r2))
    print("Average transmission rate: {}".format(len(sequence)/r1))
    print("Channel capacity:  {}".format(BSC_capacity(crossover_prob)))
    
    

# modified posterior matching scheme testing
elif test == 3: 
    sequence = "1101010"
    crossover_prob = 0.04
    block_len = 7
    max_errs = block_len
    mpms = MPMS(crossover_prob, block_len)
    # fixed rounds, always correctly decoding, no result verification
    # s, v, r = mpms.transmit(sequence, numErr=block_len, rounds=100)
    # unfixed rounds
    s, v, r = mpms.transmit(sequence, numErr=block_len)
    print("Result:\n-rounds:{}\n-real number:{}\n-binary sequence:{}".format(r,v,s))
    print("-actual sequence:{}".format(sequence))
    if s == sequence:
        print("Correct!")
    else:
        print("Wrong!")
    print("Channel capacity:  {}".format(BSC_capacity(crossover_prob)))
    print("Transmission Rate: {}".format(len(sequence)/ block_len / r))

# find out optimal channel use
elif test == 4: 
    # test paras settings
    sequence = "11011011011"
    message = int(sequence, 2) / 2**(len(sequence))
    message = 0.11011011011
    crossover_prob = 0.2
    block_len = 4 # harmming(4,7)
    max_errs = block_len
        
    # simulation settings
    min_trans, max_trans = 3, 81
    size = 100 # sample size
    ratio = []
    for t in range(min_trans, max_trans):
        count_correct = 0 # number of correct transmission
        count_total = 0 # number of successful transmission
        value = 0 # transmitted value
        for i in range(size):
            #np.random.seed(i*t)
            try:
                mpms = MPMS(crossover_prob, block_len)
                value, r = mpms.transmit(message, numErr=max_errs, rounds=t, converge=False)
                print("value is {}, {}".format(value, abs(value - message)))
                count_total += 1
                if abs(value - message) <= 1E-5:
                    count_correct += 1
                del(mpms)
            except ZeroDivisionError:
                pass # too many #TODO
            except Exception as e:
                print(e)
        count_correct /= count_total
        ratio.append(count_correct)
    
    print(ratio) 

    # plot graph
    X = np.array(range(min_trans,max_trans))
    spl = make_interp_spline(X, ratio, k=3) 
    Y = spl(X)
    plt.plot(X,Y)
    plt.xlabel("Number of channel use")
    plt.ylabel("Ratio of correct transmission")
    plt.title("Correct rate in 100 samples for different channel use")
    plt.savefig("test4.png")
    plt.show()


# find out optimal code length
elif test == 5: 
    sequence = "11011011011"
    message = int(sequence, 2) / 2**(len(sequence))
    #message = 0.73248
    crossover_prob = 0.2
    size = 100
    use = []
    for l in range(4, 14):
        max_errs = l
        count_correct = 0 # number of correct transmission
        count_total = 0 # number of successful transmission
        value = 0
        for i in range(size):
            np.random.seed(l*i)
            try:
                mpms = MPMS(crossover_prob, l)
                value, r = mpms.transmit(message, numErr=max_errs, converge=True)
                print("value is {}, {}".format(value, abs(value - message)))
                del(mpms)
                count_total += 1
                if abs(value - message) <= 1E-5:
                    count_correct += r
            except ZeroDivisionError:
                pass # TODO
            except Exception as e:
                print(e)   
        count_correct /= count_total
        use.append(count_correct)
    print(use)

    X = np.array(range(4,14))
    spl = make_interp_spline(X, use, k=3) 
    Y = spl(X)
    plt.plot(X,Y)
    plt.xlabel("Linear code length")
    plt.ylabel("Channel use")
    plt.title("Average channel use in 100 samples for different code length")
    plt.savefig("test5.png")
    plt.show()


# experiment on transmission rate
elif test == 6: 
    # test paras settings
    sequence = "11011011011"
    message = int(sequence, 2) / 2**(len(sequence))
    #message = 0.11011011011
    crossover_prob = 0.04
    block_len = 9 # harmming(4,7)
    max_errs = block_len

    # simulation settings
    size = 100 # sample size
    value = 0 # transmitted value
    trans_round = [] # number of linear code use 
    for i in range(size):
        try:
            mpms = MPMS(crossover_prob, block_len)
            value, rounds = mpms.transmit(message, numErr=max_errs, converge=True)
            print("value is {}, {}".format(value, abs(value - message)))
            if abs(value - message) <= 1E-5:
                trans_round.append(rounds)
            del(mpms)
        except ZeroDivisionError:
            pass
        except Exception as e:
            print(e)
    
    mean = np.mean(trans_round)
    med = np.median(trans_round)
    print(len(trans_round))
    print(trans_round)

    n = len(sequence)
    code_len = HammingCode.calcRedundantBits(None, block_len) + block_len
    rate_mean = n / code_len / mean
    rate_median = n / code_len / med
    capacity = BSC_capacity(crossover_prob)
    print(rate_mean, rate_median, capacity, mean, med) 





