from mpms import MPMS
from pms import PMS
from hamming import HammingCode


'''
Available Testing
'''
# hamming code testing:
# hamming = HammingCode('1001')
# print(hamming.calcRedundantBits(1))
# print(hamming. genCode())
# hamming.detectError('100', nr=2)
# print(hamming.decode('1001100'[::-1]))

# posterior matching scheme testing
# pms = PMS(0.2)
# pms.transmit(0.6)

# modified posterior matching scheme testing
mpms = MPMS(0.2, 7)
mpms.transmit(0.0314159, 1)