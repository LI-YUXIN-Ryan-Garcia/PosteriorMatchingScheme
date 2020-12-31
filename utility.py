import numpy as np

# entropy function
def h(x): return x * np.log2(1/x) + (1-x) * np.log2(1/(1-x))

# capacity
def BSC_capacity(alpha): return 1 - h(alpha)