import os
import numpy as np

min_len = 1
max_len = 200
sample_size = 1000

np.random.seed(0)
for l in range(min_len, max_len+1):
    fn = 'msg_{}.txt'.format(l)
    fn = os.path.join('message',fn)
    with open(fn, 'w') as f:
        for i in range(sample_size):
            msg = np.random.randint(2, size=l)
            f.write(''.join(map(str,msg))+'\n')
