from __future__ import division
import numpy as np
import sys
import numpy.matlib
from loadData import loadData

class Neural_Network(object):
    def __init__(self, hls, iters, l):
        #Defining Hyperparameters
        self.inputLayerSize = 784
        self.hiddenLayerSize = hls
        self.outputLayerSize = 10
        self.Lambda = l
        iterations = iters
        tol = 1e-6

        # LOAD FILE HERE
        self.Xtrain, self.Xtest, self.Ntrain, self.Ntest, self.T, self.TtestTrue = loadData()

        # Normilize pixels to take values between [0, 1]
        self.Xtrain = np.divide(self.Xtrain, 255.0)
        self.Xtest = np.divide(self.Xtest, 255.0)

        # print 'Xtrain shape: ', np.shape(self.Xtrain), '\nXtest shape: ', np.shape(self.Xtest)

        self.N1, self.D1 = np.shape(self.Xtrain)
        self.N2, self.D2 = np.shape(self.Xtest)

        # Add an extra column of ones for both the training and test examples
        self.Xtrain = np.hstack((np.ones((int(np.sum(self.Ntrain)), 1)), self.Xtrain))
        self.Xtest = np.hstack((np.ones((int(np.sum(self.Ntest)), 1)), self.Xtest))

        # List containing:
        # [0]: Maximum number of iterations of the gradient ascend
        # [1]: Tolerance
        # [2]: Learning rate
        self.options = [iterations, tol, 0.5/self.N1]

        #Weights initialization
        self.W1init = np.random.randn(self.hiddenLayerSize, self.inputLayerSize+1)/np.sqrt(self.inputLayerSize)
        self.W2init = np.random.randn(self.outputLayerSize, self.hiddenLayerSize+1)/np.sqrt(self.hiddenLayerSize)
        # print '\nW1init shape: ', np.shape(self.W1init), '\nW2init shape: ', np.shape(self.W2init)


    def costFunction(self, W1, W2, X, T, Lambda):
        # Forward Propagation
        z2 = np.dot(X, W1.T)
        a2 = np.hstack((np.ones((np.shape(z2)[0], 1)), self.activationFunction(z2)))
        z3 = np.dot(a2, W2.T)
        yhat = self.softmaxProbabilities(z3)

        # Cost function
        M = z3.max(axis=1, keepdims=1)
        Ew = np.sum((T * z3).sum(axis=0,keepdims=1)) - np.sum(M) - np.sum(np.log((np.exp(z3 - np.matlib.repmat(M,1,np.size(W2, 0)))).sum(axis=1,keepdims=1))) - (0.5*Lambda)*np.sum((W2 * W2).sum(axis=0,keepdims=1))

        # Backpropagation
        dJdW2 = np.dot((T - yhat).T, a2) - (Lambda* W2)
        dJdW1 = np.dot(np.dot(W2[:,1:].T, (T -yhat).T) * self.activationFunctionPrime(z2).T, X) - (Lambda*W1)
        return Ew, dJdW1, dJdW2


    def gradCheck(self, W1, W2, X, T, Lambda):
        K1, D1 = np.shape(W1)
        K2, D2 = np.shape(W2)

        #  Compute the analytic gradient
        Ew, gradEw1, gradEw2 = self.costFunction(W1, W2, X, T, Lambda)

        epsilon = 1e-6
        numgradEw1 = np.zeros((K1, D1))
        numgradEw2 = np.zeros((K2, D2))

        # Numerical gradient estimates for W1
        for k in range(K1):
            for d in range(D1):
                Wtmp = np.copy(W1)
                Wtmp[k,d] = Wtmp[k,d] + epsilon
                Ewplus,_,_ = self.costFunction(Wtmp, W2, X, T, Lambda)

                Wtmp = np.copy(W1)
                Wtmp[k, d] = Wtmp[k, d] - epsilon
                Ewminus,_,_ = self.costFunction(Wtmp, W2, X, T, Lambda)

                numgradEw1[k, d] = (Ewplus - Ewminus)/(2*epsilon)

        diff1 = np.absolute(gradEw1 - numgradEw1)

        # Numerical gradient estimates for W2
        for k in range(K2):
            for d in range(D2):
                Wtmp = np.copy(W2)
                Wtmp[k,d] = Wtmp[k,d] + epsilon
                Ewplus,_,_ = self.costFunction(W1, Wtmp, X, T, Lambda)

                Wtmp = np.copy(W2)
                Wtmp[k, d] = Wtmp[k, d] - epsilon
                Ewminus,_,_ = self.costFunction(W1, Wtmp, X, T, Lambda)

                numgradEw2[k, d] = (Ewplus - Ewminus)/(2*epsilon)

        diff2 = np.absolute(gradEw2 - numgradEw2)

        return diff1, diff2


    def NNTrain(self, T, Xtrain, Lambda, W1init, W2init, options):
        W1 = W1init
        W2 = W2init

        iterations = options[0]
        tol = options[1]
        eta = options[2]

        Ewold = -np.inf

        for it in range(iterations):
            Ew, gradEw1, gradEw2 = self.costFunction(W1, W2, Xtrain, T, Lambda)
            print('Iteration: {}, Cost function: {}'.format(it+1, Ew));

            if np.absolute(Ew - Ewold) < tol:
                break;

            W1 = W1 + eta*gradEw1
            W2 = W2 + eta*gradEw2

            Ewold = Ew

        return W1, W2


    def NNTest(self, W1, W2, Xtest):
        # Forward Propagation
        z2 = np.dot(Xtest, W1.T)
        a2 = np.hstack((np.ones((np.shape(z2)[0], 1)), self.activationFunction(z2)))
        z3 = np.dot(a2, W2.T)
        Yhat_test = self.softmaxProbabilities(z3)

        tmp = np.argmax(Yhat_test, 1)
        Ttest = tmp.reshape(tmp.shape[0], 1)

        return Ttest, Yhat_test


    def softmaxProbabilities(self, Y):
        N, K = np.shape(Y)
        M = Y.max(axis=1, keepdims=1)
        Y = Y - np.matlib.repmat(M, 1, K)
        Y = np.exp(Y)


        tmp = np.sum(Y,1)
        tmp = tmp.reshape(tmp.shape[0], 1)
        S = Y/np.matlib.repmat(tmp, 1, K)
        return S


    def activationFunction(self, z):
            return np.cos(z)


    def activationFunctionPrime(self, z):
            return -np.sin(z)
