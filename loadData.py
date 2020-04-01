import numpy as np

def loadData():
    Xtmp1 = []
    Xtmp2 = []
    Xtrain = []
    Xtest = []
    T = np.zeros((60000, 10))
    TtestTrue = np.zeros((10000, 10))
    Ntrain = np.zeros((10,1))
    Ntest = np.zeros((10,1))

    for j in range(10):
        Xtmp1.append('mnistdata/train%d.txt' % j)
        Xtmp2.append('mnistdata/test%d.txt' % j)

    print('\n')
    i=0
    for j in range(10):
        print('Loading file: train{}.txt'.format(j))
        with open(Xtmp1[j]) as f:
            for line in f:
                Xtrain.append(line.split())
                T[i][j] = 1
                Ntrain[j] += 1
                i += 1
            f.close()

    i=0
    for j in range(10):
        print('Loading file: test{}.txt'.format(j))
        with open(Xtmp2[j]) as f:
            for line in f:
                Xtest.append(line.split())
                TtestTrue[i][j] = 1
                Ntest[j] += 1
                i += 1
            f.close()

    Xtrain = np.asanyarray(Xtrain)
    Xtrain = Xtrain.astype(float)
    Xtest = np.asanyarray(Xtest)
    Xtest = Xtest.astype(float)
    Ntrain = np.asarray(Ntrain)
    Ntest = np.asarray(Ntest)

    return Xtrain, Xtest, Ntrain, Ntest, T, TtestTrue

# if __name__ == '__main__':
#     loadData()
