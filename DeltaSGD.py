import numpy as np
from Sigmoid import *

def DeltaSGD(W , X , D):
    alpha = 0.9

    N = 4  # or we can use the number of rows of X as the number of the training data
    for k in range(N):
        x_input = X [k , :].T
        d_output = D[k]

        v_weightedinput = np.matmul(W , x_input)
        y_output = Sigmoid(v_weightedinput)

        e_error = d_output - y_output
        delta_e = y_output * (1 - y_output) * e_error

        delta_W = alpha * delta_e * x_input

        W[0][0] = W[0][0] + delta_W[0]
        W[0][1] = W[0][1] + delta_W[1]
        W[0][2] = W[0][2] + delta_W[2]
    
    return W

if __name__ == "__main__":
    X = np.array( [[0 , 0 , 1] , [0 , 1 , 1] , [1 , 0 , 1] , [1 , 1 , 1]])
    D = np.array( [ [0] , [0] , [1] , [1]])

    W = 2 * np.random.random((1 , 3)) - 1

    for _epoch in range(20000) :
        W = DeltaSGD (W , X , D)

    N = 4
    for i in range(N):
        x = X[ i , :]
        v = np.matmul(W , x)
        y = Sigmoid(v)

        print (y)
