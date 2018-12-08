################################################################################
######################## MNIST_10_SINGLE_LAYER_NN.PY ###########################
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

else:
    print ('script is working in ipython/jupyter')

import matplotlib.pyplot as plt

import tensorflow as tf


def mnist_10_single_layer_nn(eta, mB, epochs, iterations,
                             MuIdx, MNIST,
                             doDropout, doDropConnect, dropoutPkeep,
                             tf_activation, tf_optimizer,
                             log, doShow):
    shapeMNIST = MNIST['shape']
    N = shapeMNIST['N']
    M = shapeMNIST['M']
    xTrainNorm = MNIST['xTrainNorm'].T
    xValidNorm = MNIST['xValidNorm'].T
    xTestNorm = MNIST['xTestNorm'].T

    tTrain = MNIST['tTrain'].T
    tValid = MNIST['tValid'].T
    tTest = MNIST['tTest'].T

    title = "mnist_10_single_layer_nn " \
            "(eta:%f, mB:%d, epochs:%d," \
            " dropout:%d, dropconn:%d," \
            " pkeep:%.4f, %s, %s)" % (
                eta, mB, epochs, doDropout,
                doDropConnect, dropoutPkeep,
                tf_activation.__name__, tf_optimizer.__name__)

    print(title)

    log.write(title+'\n')

    # Network architecture:
    # Single layer neural network, input layer 28*28= 784, output 10 (10 digits)
    # Output labels uses one-hot encoding

    # input layer             - X[batch, 784]
    # Fully connected         - W[784,10] + b[10]
    # One-hot encoded labels  - Y[batch, 10]

    # model
    # Y = softmax(X*W+b)
    # Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]

    # Training consists of finding good W elements. This will be handled automatically by
    # Tensorflow optimizer

    tf.reset_default_graph()  # To clear the defined variables and operations of the previous cell

    tf.set_random_seed(0)

    # Placeholder for input images, each data sample is 28x28 grayscale images
    # All the data will be stored in X - tensor, 4 dimensional matrix
    # The first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, N], name='X')
    # correct answers will go here
    Z = tf.placeholder(tf.float32, [None, M], name='Z')
    pkeep = tf.placeholder(tf.float32)


    # weights W[784, 10] - initialized with random values from normal distribution mean=0, stddev=1/N**0.5
    W = tf.Variable(tf.truncated_normal([N, M], stddev=1. / N ** 0.5), name='W')
    # biases b[10]
    T = tf.Variable(tf.zeros([M]), name='T')

    # Define model

    O = tf.matmul(X, W) + T
    if doDropout:
        O = tf.nn.dropout(O, pkeep)
    elif doDropConnect:
        O = tf.nn.dropout(O, pkeep) * pkeep
    O = tf.nn.softmax(O, name='O')

    # loss function: cross-entropy = - sum( Zi * log(Oi) )
    #                           O: the computed output vector
    #                           Z: the desired output vector

    # cross-entropy
    # log takes the log of each element, * multiplies the tensors element by element
    # reduce_mean will add all the components in the tensor
    # so here we end up with the total cross-entropy for all images in the batch
    cross_entropy = -tf.reduce_mean(Z * tf.log(O)) * mB * M  # normalized for batches of 100 images,
    # *10 because  "mean" included an unwanted division by 10

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(O, 1), tf.argmax(Z, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, learning rate = 0.005
    train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)

    # matplotlib visualization
    allweights = tf.reshape(W, [-1])
    allbiases = tf.reshape(T, [-1])

    # Initializing the variables
    init = tf.global_variables_initializer()

    train_losses = list()
    train_acc = list()
    test_losses = list()
    test_acc = list()

    saver = tf.train.Saver()

    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)

        sess.run(init)
        bestAccTest = 0
        for epoch in range(epochs + 1):

            # compute training values for visualization
            acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases],
                                               feed_dict={X: xTrainNorm, Z: tTrain, pkeep: 1})

            acc_va, loss_va = sess.run([accuracy, cross_entropy],
                                       feed_dict={X: xValidNorm, Z: tValid, pkeep: 1.0})
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy],
                                         feed_dict={X: xTestNorm, Z: tTest, pkeep: 1.0})

            printoutForm = "...[%d] train(acc:%.2f, loss:%.2f) valid(acc:%.2f, loss:%.2f) test(acc:%.2f, loss:%.2f)"

            if acc_tst > bestAccTest:
                bestAccTest = acc_tst
                acc_tst, loss_tst, wAll, TAll = sess.run([accuracy, cross_entropy, allweights, allbiases],
                                                         feed_dict={X: xTestNorm, Z: tTest, pkeep: 1.0})

                printoutForm += " <<< CUR BEST"


            printout = printoutForm % (epoch, acc_trn * 100, loss_trn, acc_va * 100, loss_va, acc_tst * 100, loss_tst)
            print(printout)
            log.write(printout+'\n')

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)

            if epoch < epochs:

                for i in range(iterations):
                    # training on batches of 100 images with 100 labels

                    muIdx = MuIdx[epoch, i]

                    xTrainBatch = xTrainNorm[muIdx]
                    tTrainBatch = tTrain[muIdx]

                    # the back-propagation training step
                    sess.run(train_step,
                             feed_dict={X: xTrainBatch, Z: tTrainBatch, pkeep:dropoutPkeep})

    sess.close()

    plot_loss_accuracy(train_losses, train_acc, test_losses, test_acc, title, 1)
    if doShow: plt.show()
###
