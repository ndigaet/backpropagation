import numpy as np

"""
Debería resolver esta práctica sin agregar más librerías externas
"""

def NotImplemented_message():
    print('###################################')
    print('Tienen que implementar esta función')
    print('###################################')
    return np.array([1, 1])

def densa_forward(X, W, b):
    #Xi)*W(i)+b(i)
    Y = np.dot(X, W)
    Y = Y + b
    return Y
    #return NotImplemented_message()

def MSE(X_true, X_pred):
    # 1/n sum ((x_true-X_pred)**2)
    # en forma iterativa
    # n = X_true.size
    # suma = 0
    # for i in range (0,n):
    #     dif = X_true[i]-X_pred[0][i]
    #     dif = dif ** 2
    #     suma = suma + dif    
    # Y = suma / n
    ax = None
    Y = (np.square(X_true - X_pred[0])).mean(axis=ax)
    return Y
    #return NotImplemented_message()

def MSE_grad(X_true, X_pred):
    n = len(X_true[0])
    error = X_true - X_pred
    gradient = -2*error/n
    return gradient
    #return NotImplemented_message()

def sigmoid(X):
    Y = 1/(1 + np.exp(-X)) 
    return Y
    #return NotImplemented_message()

def sigmoid_jac(Xin):
    #La derivada de la función sigmoidea  𝜎(𝑥)  es  𝜎(𝑥)(1−𝜎(𝑥))
    n = len(Xin[0])
    print(n)
    S = sigmoid(Xin)
    print(S)
    jacobian =  np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if(i == j):
                #print(f'S[0][{i}] = {S[0][i]}')
                #print(f'S[0][{j}] = {S[0][j]}')
                jacobian[i][j] = S[0][i] * (1 - S[0][j])
    
    return jacobian

def softmax(z):
    Y = np.exp(z[0])/sum(np.exp(z[0]))
    return Y
    #return NotImplemented_message()

def softmax_jac(Xin):    
    S = softmax(Xin)    
    S_diag = np.diag(S)    
    S_vector = S.reshape(S.shape[0],1)
    S_matrix = np.tile(S_vector,S.shape[0])    
    jacobian = S_diag - (S_matrix * np.transpose(S_matrix))
    return jacobian

def forward(X, P_true, weights):
    
    n = len(weights)
    Xi = X
    for i in range(0,n):
        Y[i] = densa_forward(Xi, weights[i])
        Xi = sigmoid(Y[i])
    
    P_est = Xi
    mse = MSE(P_true, P_est)
    A1_out = Y[0]
    A2_out = Y[1]
    
    print (P_est)
    print (mse)
    print (A1_out)
    print (A2_out)
    #(X, P_true, weights) que devuelva P_est, mse, X, A1_out, A2_out
    
    return NotImplemented_message()

def get_gradients(X, P_true, weights):
    return NotImplemented_message()