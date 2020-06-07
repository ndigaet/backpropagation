import numpy as np

"""
Debería resolver esta práctica sin agregar más librerías externas
"""

def densa_forward(X, W, b):
    return X.dot(W) + b

def MSE(X_true, X_pred):
    return ((X_true-X_pred)**2).mean()

def MSE_grad(X_true, X_pred):
    return ((((X_pred-X_true)*2)/X_true.shape[1]).T).mean(axis=1)

def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_jac(Xin):
    return np.diag((sigmoid(Xin)*(1-sigmoid(Xin))).reshape(-1))

def softmax(z):
    exps = np.exp(z)
    sums = np.sum(exps, axis=1)
    return np.divide(exps.T, sums).T

def softmax_jac(Xin):
    sm = softmax(Xin)
    return np.diag(sm.reshape(-1)) - sm.reshape(-1, 1).dot(sm.reshape(1, -1))

def forward(X, P_true, weights):
    D1_out = densa_forward(X, weights[0], weights[1])
    A1_out = sigmoid(D1_out)
    D2_out = densa_forward(A1_out, weights[2], weights[3])
    A2_out = sigmoid(D2_out)
    D3_out = densa_forward(A2_out, weights[4], weights[5])
    P_est = softmax(D3_out)
    mse = MSE(P_est, P_true)
    return P_est, mse, X, A1_out, A2_out

def get_gradients(X, P_true, weights):
    P_est, loss, D1_in, D2_in, D3_in = forward(X, P_true, weights)
    MSE_grad_out = MSE_grad(P_true, P_est)

    error_D3 = softmax_jac(D3_in.dot(weights[4]) + weights[5]).dot(MSE_grad_out)
    g_d3_w = error_D3.reshape(-1, 1).dot(D3_in).T
    
    error_A2 = weights[4].dot(error_D3)
    error_D2 = sigmoid_jac(D2_in.dot(weights[2]) + weights[3]).dot(error_A2)
    g_d2_w = error_D2.reshape(-1, 1).dot(D2_in).T
    
    error_A1 = weights[2].dot(error_D2)
    error_D1 = sigmoid_jac(D1_in.dot(weights[0]) + weights[1]).dot(error_A1)
    g_d1_w = error_D1.reshape(-1, 1).dot(D1_in).T
    return (g_d1_w, error_D1.reshape(-1), g_d2_w, error_D2.reshape(-1), g_d3_w, error_D3.reshape(-1)), loss