################################################################################
##################################### MAIN.PY ##################################
################################################################################



def ipython_info():
    import sys
    ip = False
    if 'ipykernel' in sys.modules:
        ip = True  # 'notebook'
    # elif 'IPython' in sys.modules:
    #    ip = 'terminal'
    return ip


ISIPYTHON = ipython_info()
DEBUG = 1

if not ISIPYTHON:
    print('working in file: ', __file__)
    import matplotlib

    # matplotlib.use('tkagg')
    matplotlib.use('agg')

    from common import *
    from loadMNIST import *

    from mnist_10_single_layer_nn import mnist_10_single_layer_nn
    from mnist_20_5_layer_nn import mnist_20_5_layer_nn
    from mnist_30_3_layer_cnn_1_layer_nn import mnist_30_3_layer_cnn_1_layer_nn

else:
    print ('script is working in ipython/jupyter')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing



MNIST = get_MNIST()

shapeMNIST = MNIST['shape']
pTr = shapeMNIST['pTr']
pTe = shapeMNIST['pTe']
pVa = shapeMNIST['pVa']
N = shapeMNIST['N']
M = shapeMNIST['M']
print (shapeMNIST)

def run_all_mnist_10_single_layer_nn(log=None):

    doClose = False
    if log is None:
        doClose = True
        log = open('logs/run_all_mnist_10_single_layer_nn.log.txt','w')

    mBs = np.r_[32, 64, 128, 256, 512]
    doDropouts = [False, True, False]
    doDropConnects = [False, False, True]
    tf_activations = [tf.nn.sigmoid, tf.nn.relu]
    tf_optimizers = [tf.train.GradientDescentOptimizer, tf.train.AdamOptimizer]

    for doDropout, doDropConnect in zip(doDropouts, doDropConnects):
        for tf_activation in tf_activations:
            for tf_optimizer in tf_optimizers:
                for mB in mBs:
                    pkeeps = [0.75, 1. - 1. / N]
                    for pkeep in pkeeps:
                        iterations = pTr // mB
                        epochs = 100
                        eta = 1. / N

                        print ("mB:%d, pTr:%d, iterations:%d\n" % (mB, pTr, iterations))

                        MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

                        p = multiprocessing.Process(target=mnist_10_single_layer_nn,
                                                    args=(eta, mB, epochs, iterations,
                                                          MuIdx, MNIST,
                                                          doDropout, doDropConnect, pkeep,
                                                          tf_activation, tf_optimizer,
                                                          log,True
                                                          ))
                        p.start()
                        p.join()
    if doClose:
        log.close()

def run_all_mnist_20_5_layer_nn(log=None):

    doClose = False
    if log is None:
        doClose = True
        log = open('logs/run_all_mnist_20_5_layer_nn.log.txt','w')

    MNIST2D = get_MNIST(flatten=False)

    shapeMNIST = MNIST2D['shape']
    pTr = shapeMNIST['pTr']
    pTe = shapeMNIST['pTe']
    pVa = shapeMNIST['pVa']
    N = shapeMNIST['N']
    M = shapeMNIST['M']
    print (shapeMNIST)

    mBs = np.r_[32, 64, 128, 256, 512]
    shapeHLs = np.c_[[512,256,128,64], [256,128,64,32]]
    doDropouts = [False, True, False]
    doDropConnects = [False, False, True]
    tf_activations = [tf.nn.sigmoid, tf.nn.relu]
    tf_optimizers = [tf.train.GradientDescentOptimizer, tf.train.AdamOptimizer]

    for doDropout, doDropConnect in zip(doDropouts, doDropConnects):
        for tf_activation in tf_activations:
            for tf_optimizer in tf_optimizers:
                for shapeHL in shapeHLs:
                    for mB in mBs:
                        pkeeps = [[0.75 for _ in shapeHL], 1. - 1./shapeHL]
                        for pkeep in pkeeps:

                            iterations = pTr // mB
                            epochs = 100
                            eta = 1. / N

                            print ("mB:%d, pTr:%d, iterations:%d\n" % (mB, pTr, iterations))

                            MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

                            p = multiprocessing.Process(target=mnist_20_5_layer_nn,
                                                        args=(eta, mB, epochs, iterations,
                                                              MuIdx, MNIST2D,
                                                              shapeHL,
                                                              doDropout, doDropConnect, pkeep,
                                                              tf_activation, tf_optimizer,
                                                              log,True
                                                              ))
                            p.start()
                            p.join()
    if doClose:
        log.close()


def run_all_mnist_30_3_layer_cnn_1_layer_nn(log=None):

    doClose = False
    if log is None:
        doClose = True
        log = open('logs/run_all_mnist_30_3_layer_cnn_1_layer_nn.log.txt', 'w')

    MNIST2D = get_MNIST(flatten=False)

    shapeMNIST = MNIST2D['shape']
    pTr = shapeMNIST['pTr']
    pTe = shapeMNIST['pTe']
    pVa = shapeMNIST['pVa']
    N = shapeMNIST['N']
    M = shapeMNIST['M']
    print (shapeMNIST)

    mBs = np.r_[32, 64, 128, 256, 512]
    shapeHLs = [
        [[20, 40, 128], [150]],
        [[20, 40, 50], [150]],
    ]
    doDropouts = [False, True, False]
    doDropConnects = [False, False, True]
    tf_activations = [tf.nn.sigmoid, tf.nn.relu]
    tf_optimizers = [tf.train.GradientDescentOptimizer, tf.train.AdamOptimizer]

    for doDropout, doDropConnect in zip(doDropouts, doDropConnects):
        for tf_activation in tf_activations:
            for tf_optimizer in tf_optimizers:
                for shapeCL, shapeFL in shapeHLs:
                    for mB in mBs:
                        pkeeps = [0.75, 1. - 1. / shapeFL[-1]]
                        for pkeep in pkeeps:
                            iterations = pTr // mB
                            epochs = 100
                            eta = 1. / (N * N)

                            print ("mB:%d, pTr:%d, iterations:%d\n" % (mB, pTr, iterations))

                            MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

                            p = multiprocessing.Process(target=mnist_30_3_layer_cnn_1_layer_nn,
                                                        args=(eta, mB, epochs, iterations,
                                                              MuIdx, MNIST2D,
                                                              shapeCL, shapeFL,
                                                              doDropout, doDropConnect, pkeep,
                                                              tf_activation, tf_optimizer,
                                                              log,True
                                                              ))
                            p.start()
                            p.join()

    if doClose:
        log.close()


def run_one_mnist_10_single_layer_nn():
    mB = 32
    iterations = pTr // mB
    validationPeriod = 100
    epochs = 100
    eta = 1. / 784
    print ("mB:%d, pTr:%d, iterations:%d\n" % (mB, pTr, iterations))

    MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

    import tensorflow as tf

    doShow = True

    mB = 32
    doDropout = False
    dropoutPkeep = 0.75
    doDropConnect = True
    tf_activation = tf.nn.relu
    tf_optimizer = tf.train.AdamOptimizer

    log = open('logs/mnist_10_single_layer_nn.mB_32.dropconn.relu.adam.log.txt', 'w')

    if 0:
        mnist_10_single_layer_nn(eta, mB, epochs, iterations,
                                 MuIdx, MNIST,
                                 doDropout, doDropConnect, dropoutPkeep,
                                 tf_activation, tf_optimizer,
                                 log, doShow)
    else:

        p = multiprocessing.Process(target=mnist_10_single_layer_nn,
                                    args=(eta, mB, epochs, iterations,
                                          MuIdx, MNIST,
                                          doDropout, doDropConnect, dropoutPkeep,
                                          tf_activation, tf_optimizer,
                                          log, doShow))
        p.start()
        p.join()
    log.close()

def run_one_mnist_20_5_layer_nn():
    mB = 32
    iterations = pTr // mB
    validationPeriod = 100
    epochs = 100
    eta = 1. / 784
    print ("mB:%d, pTr:%d, iterations:%d\n" % (mB, pTr, iterations))

    MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

    import tensorflow as tf

    doShow = True

    mB = 32
    doDropout = False
    dropoutPkeep = 0.75
    doDropConnect = True
    tf_activation = tf.nn.relu
    tf_optimizer = tf.train.AdamOptimizer

    from mnist_20_5_layer_nn import mnist_20_5_layer_nn

    log = open('logs/mnist_20_5_layer_nn.mB_32.HL_512_64.dropconn.relu.adam.log.txt', 'w')
    doShow = True
    shapeHL = [512, 256, 128, 64]

    if 0:
        mnist_20_5_layer_nn(eta, mB, epochs, iterations,
                            MuIdx, MNIST,
                            shapeHL,
                            doDropout, doDropConnect, dropoutPkeep,
                            tf_activation, tf_optimizer,
                            log, doShow)
    else:

        p = multiprocessing.Process(target=mnist_20_5_layer_nn,
                                    args=(eta, mB, epochs, iterations,
                                          MuIdx, MNIST, shapeHL,
                                          doDropout, doDropConnect, dropoutPkeep,
                                          tf_activation, tf_optimizer,
                                          log, doShow))
        p.start()
        p.join()
    log.close()


def run_one_mnist_30_3_layer_cnn_1_layer_nn():
    MNIST = get_MNIST(flatten=False)

    shapeMNIST = MNIST['shape']
    pTr = shapeMNIST['pTr']
    pTe = shapeMNIST['pTe']
    pVa = shapeMNIST['pVa']
    N = shapeMNIST['N']
    M = shapeMNIST['M']
    print (shapeMNIST)
    doShow = True

    mB = 32
    iterations = pTr // mB
    epochs = 100
    eta = 1. / 784
    print ("mB:%d, pTr:%d, iterations:%d\n" % (mB, pTr, iterations))
    MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

    doDropout = False
    dropoutPkeep = 0.75
    doDropConnect = True
    tf_activation = tf.nn.relu
    tf_optimizer = tf.train.AdamOptimizer

    from mnist_30_3_layer_cnn_1_layer_nn import mnist_30_3_layer_cnn_1_layer_nn

    log = open('logs/mnist_30_3_layer_cnn_1_layer_nn.mB_32.CL_20_40_128.FL_150.dropconn.relu.adam.log.txt', 'w')
    doShow = True
    shapeCL, shapeFL = [20, 40, 128], [150]
    if 0:
        mnist_30_3_layer_cnn_1_layer_nn(eta, mB, epochs, iterations,
                                        MuIdx, MNIST,
                                        shapeCL, shapeFL,
                                        doDropout, doDropConnect, dropoutPkeep,
                                        tf_activation, tf_optimizer,
                                        log, doShow)
    else:

        p = multiprocessing.Process(target=mnist_30_3_layer_cnn_1_layer_nn,
                                    args=(eta, mB, epochs, iterations,
                                          MuIdx, MNIST, shapeCL, shapeFL,
                                          doDropout, doDropConnect, dropoutPkeep,
                                          tf_activation, tf_optimizer,
                                          log, doShow))
        p.start()
        p.join()
    log.close()


if __name__ == '__main__':
    print('run mnist main')
    run_one_mnist_10_single_layer_nn()
    run_one_mnist_20_5_layer_nn()
    run_one_mnist_30_3_layer_cnn_1_layer_nn()

    if 0:
        log = open('logs/mnist.alltests.log.txt','w')
        run_all_mnist_10_single_layer_nn(log)
        run_all_mnist_20_5_layer_nn(log)
        run_all_mnist_30_3_layer_cnn_1_layer_nn(log)
        log.close()