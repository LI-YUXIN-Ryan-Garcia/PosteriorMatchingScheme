'''
Implement of Hamming Code
'''     

class HammingCode():

    def __init__(self, data, receive=None):
        self.X = data
        self.Y = receive
        self.r = self.calcRedundantBits()

    def calcRedundantBits(self,m=None): 
        if m is None:
            x = self.X
            m = len(x)
        i = 0
        while 2**i < m + i + 1:
            i += 1
        return i 

    def genCode(self):
        msg = self.posRedundantBits()
        msg = self.calcParityBits(msg)
        self.code = msg
        return msg

    def posRedundantBits(self):
        data = self.X 
        r = self.r
        j = 0
        k = 1
        m = len(data) 
        res = '' 

        for i in range(1, m + r+1): 
            if i == 2**j: 
                res = res + '0'
                j += 1
            else: 
                res = res + data[-1 * k] 
                k += 1

        return res[::-1] 

    def calcParityBits(self, arr): 
        r = self.r
        n = len(arr) 

        for i in range(r): 
            val = 0
            for j in range(1, n + 1): 
                if j & (2**i) == (2**i): 
                    val = val ^ int(arr[-1 * j]) 
                    # -1 * j is given since array is reversed 

            # String Concatenation 
            # (0 to n - 2^r) + parity bit + (n - 2^r + 1 to n) 
            arr = arr[:n-(2**i)] + str(val) + arr[n-(2**i)+1:] 
        return arr 

    def detectError(self, arr, nr=None): 
        if nr is None:
            nr = self.r
        
        n = len(arr) 
        res = 0 # binary sequence

        for i in range(nr): 
            val = 0
            for j in range(1, n + 1): 
                if j & (2**i) == (2**i): 
                    val = val ^ int(arr[-1 * j])
                    # print(-1 * j)
                    # print('val = {}'.format(val)) 

            res = res + val*(10**i)
            # print(res) 

        # Convert binary to decimal 
        err_pos = int(str(res), 2)
        print('Error position: {}'.format(err_pos))
        return err_pos

    def decode(self, c=None):
        if c is None:
            c = self.code[::-1]
        
        m = ''    
        j = 0
        for i in range(1, len(c)+1):
            if i == 2**j:
                j += 1
            else:
                m += c[i-1]
        return m[::-1]