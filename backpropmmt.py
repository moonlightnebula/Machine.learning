import numpy as np
from Sigmoid import *

def BackpropMmt(W1 , W2 , X , D):
    alpha = 0.9
    beta = 0.9

    mmt_1 = np.zeros_like(W1)
    mmt_2 = np.zeros_like(W2)
    
    N = 4  # or we can use the number of rows of X as the number of the training data
    for k in range(N):
        x_input = X [k , :].T
        d_output = D[k]
        
        # print (np.size(x_input))
        # print (np.size(d_output))

        v_1 = np.matmul( W1 , x_input)
        y_1 = Sigmoid(v_1)

        # print (np.size(v_1) ,np.size( y_1))

        v_output = np.matmul( W2 , y_1)
        y_output = Sigmoid(v_output)

        # print (v , y_output)
        
        e_final = d_output - y_output
        delta = y_output * (1 - y_output) * e_final

        # print ("e , delta = " , e , delta)

        e_1 = np.matmul(W2.T , delta)
        delta_1 = y_1 * (1 - y_1) * e_1

        # print("e1 , delta1 =" , e_1 , delta_1)

        dW_1 = (alpha*delta_1).reshape(4, 1) * x_input.reshape(1, 3)
        mmt_1 = dW_1 + beta * mmt_1
        W1 = W1 + mmt_1
        # print ("dw1 , W1 " , dW_1 , W1 )
        dW_2 = alpha * delta * y_1
        mmt_2 = dW_2 + beta * mmt_2
        W2 = W2 + mmt_2
        # print ("dw2 , W2 " , np.size(dW_2) , np.size(W2) )


    return (W1 , W2)

if __name__ == "__main__":
    X = np.array( [[0 , 0 , 1] , [0 , 1 , 1] , [1 , 0 , 1] , [1 , 1 , 1]])
    D = np.array( [ [0] , [1] , [1] , [0]])

    W1 = 2 * np.random.random((4 , 3)) - 1
    W2 = 2 * np.random.random((1 , 4)) -1 

    for epoch in range(40000):
        W1 , W2 = BackpropMmt(W1 , W2 , X , D)

    N = 4
    for k in range(N):
        x_input = X [k , :].T
        v1 = np.matmul(W1, x_input)
        y1 = Sigmoid(v1)
        v  = np.matmul(W2, y1)
        y  = Sigmoid(v)
        print(y)


