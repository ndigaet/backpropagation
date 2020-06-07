import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD

def get_deep_model(lr=0.1, layers=10, input_dim=2):
    model = Sequential()
    model.add(Dense(3, name='D1', input_dim=input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.5), 
                    bias_initializer=initializers.RandomNormal(stddev=0.5)))
    model.add(Activation('sigmoid', name='A1'))
    for i in range(layers-2):
        model.add(Dense(2, name=f'D{i+2}', kernel_initializer=initializers.RandomNormal(stddev=0.5), 
                        bias_initializer=initializers.RandomNormal(stddev=0.5)))
        model.add(Activation('sigmoid', name=f'A{i+2}'))
    model.add(Dense(3,  name=f'D{i+3}', kernel_initializer=initializers.RandomNormal(stddev=0.5), 
                    bias_initializer=initializers.RandomNormal(stddev=0.5)))
    model.add(Activation('softmax', name='P_est'))
    model.compile(SGD(lr=lr), loss='mse')
    model.save('deep_model.hdf5')
    return model

def get_model(lr=0.1, input_dim=2):
    model = Sequential()
    model.add(Dense(3, name='D1', input_dim=input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.5), 
                    bias_initializer=initializers.RandomNormal(stddev=0.5)))
    model.add(Activation('sigmoid', name='A1'))
    model.add(Dense(2, name='D2', kernel_initializer=initializers.RandomNormal(stddev=0.5), 
                    bias_initializer=initializers.RandomNormal(stddev=0.5)))
    model.add(Activation('sigmoid', name='A2'))
    model.add(Dense(3,  name='D3', kernel_initializer=initializers.RandomNormal(stddev=0.5), 
                    bias_initializer=initializers.RandomNormal(stddev=0.5)))
    model.add(Activation('softmax', name='P_est'))
    model.compile(SGD(lr=lr), loss='mse')
    model.save('simple_model.hdf5')
    return model

def densa_forward(X, W, b):
    return X.dot(W) + b

def MSE(X_true, X_pred):
    return ((X_true-X_pred)**2).mean()

def MSE_grad(X_true, X_pred):
    return (((X_pred-X_true)*2)/X_true.shape[1]).T

def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_jac(Xin):
    return np.diag((sigmoid(Xin)*(1-sigmoid(Xin))).reshape(-1))

def softmax(z):
    exps = np.exp(z)
    sums = np.sum(exps)
    return np.divide(exps, sums)

def softmax_jac(Xin):
    sm = softmax(Xin)
    return np.diag(sm.reshape(-1)) - sm.reshape(-1, 1).dot(sm.reshape(1, -1))

