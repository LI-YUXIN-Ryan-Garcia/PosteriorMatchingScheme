from mpms import MPMS
from pms import PMS
from hamming import HammingCode
import numpy as np

'''
Available Testing
'''
test = 3
np.random.seed(0)

if test == 1: # hamming code testing:
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
    

elif test == 2: # standard posterior matching scheme testing
    pms = PMS(0.2)
    pms.transmit(0.6)

elif test == 3: # modified posterior matching scheme testing
    # message = 0.0314159
    message = 0.73248
    error_probability = 0.3
    block_len = 7
    num_of_errs = block_len
    mpms = MPMS(message, block_len)
    mpms.transmit(message, rounds=100, converge=False)

elif test == 4: # num of transmission
    message = 0.0314159
    error_probability = 0.3
    block_len = 7
    num_of_errs = block_len
    rounds = 40
    size = 100 # sample size
    ratios = []
    for t in range(20,rounds):
        count = 0
        value = 0
        for i in range(size):
            try:
                mpms = MPMS(message, block_len)
                value, r = mpms.transmit(message, rounds=t, converge=False)
                del(mpms)
            except ZeroDivisionError:
                pass
            except Exception as e:
                print(e)
            if abs(value - message) <= 1E-6:
                count += 1
        count /= size
        ratios.append(count)
    print(ratios)





