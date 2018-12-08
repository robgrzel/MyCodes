################################################################################
############################# MNIST_20_5_LAYER_NN.PY ###########################
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


def mnist_20_5_layer_nn(eta, mB, epochs, iterations,
                        MuIdx, MNIST,
                        shapeHL,
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

    title = "mnist_20_5_layer_nn " \
            "(eta:%f, mB:%d, epochs:%d," \
            " shape:%s, dropout:%d, dropconn:%d," \
            " pkeep:%.4f, %s, %s)" % (
                eta, mB, epochs, shapeHL,
                doDropout,doDropConnect, dropoutPkeep,
                tf_activation.__name__, tf_optimizer.__name__)

    print(title)

    log.write(title+'\n')

    # Network architecture:
    # Five layer neural network, input layer 28*28= 784, output 10 (10 digits)
    # Output labels uses one-hot encoding

    # input layer             - X[batch, 784]
    # 1 layer                 - W1[784, 200] + b1[200]
    #                           Y1[batch, 200]
    # 2 layer                 - W2[200, 100] + b2[100]
    #                           Y2[batch, 200]
    # 3 layer                 - W3[100, 60]  + b3[60]
    #                           Y3[batch, 200]
    # 4 layer                 - W4[60, 30]   + b4[30]
    #                           Y4[batch, 30]
    # 5 layer                 - W5[30, 10]   + b5[10]
    # One-hot encoded labels    Y5[batch, 10]

    # model
    # Y = softmax(X*W+b)
    # Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]

    # Training consists of finding good W elements. This will be handled automaticaly by
    # Tensorflow optimizer

    tf.reset_default_graph()  # To clear the defined variables and operations of the previous cell

    tf.set_random_seed(0)

    # mnist.test (10K images+labels) -> mnist.test.images, mnist.test.labels
    # mnist.train (60K images+labels) -> mnist.train.images, mnist.test.labels

    # Placeholder for input images, each data sample is 28x28 grayscale images
    # All the data will be stored in X - tensor, 4 dimensional matrix
    # The first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, N], name='X')
    # correct answers will go here
    Z = tf.placeholder(tf.float32, [None, M], name='Z')
    pkeep = tf.placeholder(tf.float32)

    # layers sizes
    L0 = N
    L1 = shapeHL[0]
    L2 = shapeHL[1]
    L3 = shapeHL[2]
    L4 = shapeHL[3]
    L5 = M

    # weights - initialized with random values from normal distribution mean=0, stddev=0.1
    # output of one layer is input for the next
    W1 = tf.Variable(tf.truncated_normal([L0, L1], stddev=1. / L0 ** 0.5), name='W1')
    T1 = tf.Variable(tf.zeros([L1]), name='T1')

    W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=1. / L1 ** 0.5), name='W2')
    T2 = tf.Variable(tf.zeros([L2]), name='T2')

    W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=1. / L2 ** 0.5), name='W3')
    T3 = tf.Variable(tf.zeros([L3]), name='T3')

    W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=1. / L3 ** 0.5), name='W4')
    T4 = tf.Variable(tf.zeros([L4]), name='T4')

    W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=1. / L4 ** 0.5), name='W5')
    T5 = tf.Variable(tf.zeros([L5]), name='T5')

    # Define model

    V1 = tf_activation(tf.matmul(X, W1) + T1)
    if doDropout:
        V1 = tf.nn.dropout(V1, pkeep)
    elif doDropConnect:
        V1 = tf.nn.dropout(V1, pkeep) * pkeep

    V2 = tf_activation(tf.matmul(V1, W2) + T2)
    if doDropout:
        V2 = tf.nn.dropout(V2, pkeep)
    elif doDropConnect:
        V2 = tf.nn.dropout(V2, pkeep) * pkeep

    V3 = tf_activation(tf.matmul(V2, W3) + T3)
    if doDropout:
        V3 = tf.nn.dropout(V3, pkeep)
    elif doDropConnect:
        V3 = tf.nn.dropout(V3, pkeep) * pkeep

    V4 = tf_activation(tf.matmul(V3, W4) + T4)
    if doDropout:
        V4 = tf.nn.dropout(V4, pkeep)
    elif doDropConnect:
        V4 = tf.nn.dropout(V4, pkeep) * pkeep

    b5 = tf.matmul(V4, W5) + T5
    O = tf.nn.softmax(b5)

    # loss function: cross-entropy = - sum( Z * log(O) )
    #                           O: the computed output vector
    #                           Z: the desired output vector

    # cross-entropy
    # log takes the log of each element, * multiplies the tensors element by element
    # reduce_mean will add all the components in the tensor
    # so here we end up with the total cross-entropy for all images in the batch
    # cross_entropy = -tf.reduce_mean(Z * tf.log(O)) * mB  # normalized for batches of mB images,

    # we can also use tensorflow function for softmax
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=b5, labels=Z)
    cross_entropy = tf.reduce_mean(cross_entropy) * mB

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(O, 1), tf.argmax(Z, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, learning rate = 0.005
    train_step = tf_optimizer(eta).minimize(cross_entropy)

    # matplotlib visualisation
    allweights = tf.concat([tf.reshape(W1, [-1]),
                            tf.reshape(W2, [-1]),
                            tf.reshape(W3, [-1]),
                            tf.reshape(W4, [-1]),
                            tf.reshape(W5, [-1])], 0)
    allbiases = tf.concat([tf.reshape(T1, [-1]),
                           tf.reshape(T2, [-1]),
                           tf.reshape(T3, [-1]),
                           tf.reshape(T4, [-1]),
                           tf.reshape(T5, [-1])], 0)

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
                                               feed_dict={X: xTrainNorm, Z: tTrain, pkeep: 1.0})

            acc_va, loss_va = sess.run([accuracy, cross_entropy],
                                       feed_dict={X: xValidNorm, Z: tValid, pkeep: 1.0})
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy],
                                         feed_dict={X: xTestNorm, Z: tTest, pkeep: 1.0})

            printoutForm = "...[%d] train(acc:%.2f, loss:%.2f) valid(acc:%.2f, loss:%.2f) test(acc:%.2f, loss:%.2f)"

            if acc_tst > bestAccTest:
                bestAccTest = acc_tst
                saver.save(sess, "checkpoint/%s" % title)
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
                             feed_dict={X: xTrainBatch, Z: tTrainBatch, pkeep: dropoutPkeep})

    plot_loss_accuracy(train_losses, train_acc, test_losses, test_acc, title, 1)
    if doShow: plt.show()


#######
