from __future__ import division
import numpy as np
import Neural_Network as nn

# Initialization of the Neural Network
# Neural_Network(hiddenLayerSize, Iterations, Lambda)
hidden_units = 500
iterations = 500
Lambda = 0.1
a = nn.Neural_Network(400, 500, 0.1)


# Gradient Check
grad_check = input('\nExecute gradient checking? (y|n) ')
print('\n')
error = 1e-6


if grad_check == 'yes' or grad_check == 'y':
    ch = np.random.permutation(a.N1)
    ch = ch[:10]
    print(ch)

    diff1, diff2 = a.gradCheck(a.W1init, a.W2init, a.Xtrain[ch,:], a.T[ch,:], a.Lambda)
    # if (diff1 < error).all() and (diff2 < error).all():
    #     print 'Gradient Check Passed!'
    # else:
    #     print 'Gradient Check Failed'


# Model training
W1, W2 = a.NNTrain(a.T, a.Xtrain, a.Lambda, a.W1init, a.W2init, a.options)


# Model Testing
Ttest, Yhat_test = a.NNTest(W1, W2, a.Xtest)


tmp = np.argmax(a.TtestTrue, 1)
Ttrue = tmp.reshape(tmp.shape[0], 1)

err = np.sum(Ttest!=Ttrue)/10000.0
print('\nThe error of the method for h_units={}, iters={} and lambda={} is : '.format(hidden_units, iterations, Lambda), err, '\n')
