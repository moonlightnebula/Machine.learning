import numpy as np
from Sigmoid import *

def BackPropCe(W1 , W2 , X , D):
    alpha = 0.9

    N = 4  # or we can use the number of rows of X as the number of the training data
    for k in range(N):
        x_input = X [k , :].T
        d_output = D[k]

        v_1 = np.matmul( W1 , x_input)
        y_1 = Sigmoid(v_1)

        v = np.matmul( W2 , y_1)
        y_output = Sigmoid(v)

        e = d_output - y_output
        delta = e

        e_1 = np.matmul(W2.T , delta)
        delta_1 = y_1 * (1 - y_1) * e_1

        dW_1 = (alpha*delta_1).reshape(4, 1) * x_input.reshape(1, 3)
        W1 = W1 + dW_1

        dW_2 = alpha * delta * y_1
        W2 = W2 + dW_2

    return (W1 , W2)

if __name__ == "__main__":
    X = np.array( [[0 , 0 , 1] , [0 , 1 , 1] , [1 , 0 , 1] , [1 , 1 , 1]])
    D = np.array( [ [0] , [1] , [1] , [0]])

    W1 = 2 * np.random.random((4 , 3)) - 1
    W2 = 2 * np.random.random((1 , 4)) -1 

    for epoch in range(10000):
        W1 , W2 = BackPropCe(W1 , W2 , X , D)

    N = 4
    for k in range(N):
        x_input = X [k , :].T
        v1 = np.matmul(W1, x_input)
        y1 = Sigmoid(v1)
        v  = np.matmul(W2, y1)
        y  = Sigmoid(v)
        print(y)


