from pms import PMS
from hamming import HammingCode
from utility import h, BSC_capacity
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import statistics as stat


# parameter
# bit_sequence = '1001' # length 4
# bit_sequence = '110110110110111001000100011100110001' # length 36
bit_sequence = '11010111010101110101001010100010101101001010101110010101000100100101000100101001'[:53] # length 80
crossover_probability = 0.2
error_probability = 0.01

def test_pms(sequence, crossover_prob, error_prob):
    pms = PMS(crossover_prob)
    s, v, r = pms.transmit(sequence, rounds=1000)
    print("Result:\n-rounds:{}\n-real number:{}\n-binary sequence:{}".format(r,v,s))
    print("-actual sequence:{} len: {}".format(sequence, len(sequence)))
    if s == sequence:
        print("Correct!")
    else:
        print("Wrong!")
    print("Transmission Rate: {}".format(len(sequence) / r))
    print("Channel capacity:  {}".format(BSC_capacity(crossover_prob)))


test_pms(bit_sequence, crossover_probability, error_probability)