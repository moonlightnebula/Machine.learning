import numpy as np

def Sigmoid(v):

    out_put = 1 / (1 + np.exp(-v))
    return out_put

if __name__ == '__main__':
    print (Sigmoid(0))