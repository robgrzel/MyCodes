################################################################################
################# MNIST_30_3_LAYER_CNN_1_LAYER_NN.PY ###########################
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

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def mnist_30_3_layer_cnn_1_layer_nn(eta, mB, epochs, iterations,
                                    MuIdx, MNIST2D,
                                    shapeCL, shapeFL,
                                    doDropout, doDropConnect, dropoutPkeep,
                                    tf_activation, tf_optimizer,
                                    log, doShow):
    shapeMNIST = MNIST2D['shape']
    N = shapeMNIST['N']
    M = shapeMNIST['M']
    xTrainNorm = MNIST2D['xTrainNorm'].T
    xValidNorm = MNIST2D['xValidNorm'].T
    xTestNorm = MNIST2D['xTestNorm'].T

    tTrain = MNIST2D['tTrain'].T
    tValid = MNIST2D['tValid'].T
    tTest = MNIST2D['tTest'].T

    title = "mnist_30_3_layer_cnn_1_layer_nn " \
            "(eta:%f, mB:%d, epochs:%d," \
            " shape:%s_%s, dropout:%d, dropconn:%d," \
            " pkeep:%.4f, %s, %s)\n" % (
                eta, mB, epochs, shapeCL, shapeFL,
                doDropout, doDropConnect, dropoutPkeep,
                tf_activation.__name__, tf_optimizer.__name__)


    print(title)

    log.write(title+'\n')

    # Network architecture:
    # 5 layer neural network with 3 convolution layers, input layer 28x28x1, output 10 (10 digits)
    # Output labels uses one-hot encoding

    # input layer               - X[batch, 28, 28]
    # 1 conv. layer             - W1[5,5,,1,C1] + b1[C1]
    #                             Y1[batch, 28, 28, C1]
    # 2 conv. layer             - W2[3, 3, C1, C2] + b2[C2]
    # 2.1 max pooling filter 2x2, stride 2 - down sample the input (rescale input by 2) 28x28-> 14x14
    #                             Y2[batch, 14,14,C2]
    # 3 conv. layer             - W3[3, 3, C2, C3]  + b3[C3]
    # 3.1 max pooling filter 2x2, stride 2 - down sample the input (rescale input by 2) 14x14-> 7x7
    #                             Y3[batch, 7, 7, C3]
    # 4 fully connecteed layer  - W4[7*7*C3, FC4]   + b4[FC4]
    #                             Y4[batch, FC4]
    # 5 output layer            - W5[FC4, 10]   + b5[10]
    # One-hot encoded labels      Y5[batch, 10]

    # Training consists of finding good W_i elements. This will be handled automatically by
    # Tensorflow optimizer

    tf.reset_default_graph()  # To clear the defined variables and operations of the previous cell

    tf.set_random_seed(0)

    # mnist.test (10K images+labels) -> mnist.test.images, mnist.test.labels
    # mnist.train (60K images+labels) -> mnist.train.images, mnist.test.labels

    # Placeholder for input images, each data sample is 28x28 grayscale images
    # All the data will be stored in X - tensor, 4 dimensional matrix
    # The first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, N, N, 1], name='X')
    # correct answers will go here
    Z = tf.placeholder(tf.float32, [None, M], name='Z')
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)

    # layers sizes
    L0 = N
    C1 = shapeCL[0]  # 20 first convolutional layer output depth (num filters)
    C2 = shapeCL[1]  # 40 second convolutional layer output depth (num filters)
    C3 = shapeCL[2]  # 128 third convolutional layer output depth (num filters)
    FC4 = shapeFL[0]  # 150 fully connected layer
    L5 = M

    # weights - initialized with random values from normal distribution mean=0, stddev=0.1
    # 5x5 conv. window, 1 input channel (gray images), C1 - outputs
    W1 = tf.Variable(tf.truncated_normal([4, 4, 1, C1], stddev=1. / np.prod([4, 4, 1, ]) ** 0.5), name='W1')
    T1 = tf.Variable(tf.zeros([C1]), name='T1')
    # 3x3 conv. window, C1 input channels(output from previous conv. layer ), C2 - outputs
    W2 = tf.Variable(tf.truncated_normal([5, 5, C1, C2], stddev=1. / np.prod([5, 5, C1, ]) ** 0.5), name='W2')
    T2 = tf.Variable(tf.zeros([C2]), name='T2')
    # 3x3 conv. window, C2 input channels(output from previous conv. layer ), C3 - outputs
    W3 = tf.Variable(tf.truncated_normal([3, 3, C2, C3], stddev=1. / np.prod([3, 3, C2, ]) ** 0.5), name='W3')
    T3 = tf.Variable(tf.zeros([C3]), name='T3')
    # fully connected layer, we have to reshpe previous output to one dim,
    # we have two max pool operation in our network design,
    # so our initial size 28x28 will be reduced to 14x14, then 7x7
    # (here each max poll will reduce size by factor of k)
    W4 = tf.Variable(tf.truncated_normal([7 * 7 * C3, FC4], stddev=1. / np.prod([7, 7, C3, ]) ** 0.5), name='W4')
    T4 = tf.Variable(tf.zeros([FC4]), name='T4')
    # output softmax layer (10 digits)
    W5 = tf.Variable(tf.truncated_normal([FC4, L5], stddev=1. / FC4 ** 0.5), name='W5')
    T5 = tf.Variable(tf.zeros([L5]), name='T5')

    # Define model
    stride = 1  # output is 28x28
    V1 = tf_activation(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + T1)
    # V1[batch, 28, 28, C1]
    k = 2  # max pool filter size and stride, will reduce input by factor of 2
    V2 = tf_activation(tf.nn.conv2d(V1, W2, strides=[1, stride, stride, 1], padding='SAME') + T2)
    V2 = tf.nn.max_pool(V2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    # V2[batch, 14,14,C2]
    V3 = tf_activation(tf.nn.conv2d(V2, W3, strides=[1, stride, stride, 1], padding='SAME') + T3)
    V3 = tf.nn.max_pool(V3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    # V3[batch, 7, 7, C3]
    # reshape the output from the third convolution for the fully connected layer
    OO = tf.reshape(V3, shape=[-1, 7 * 7 * C3])
    # OO[batch, 7*7*C3]
    V4 = tf_activation(tf.matmul(OO, W4) + T4)

    tf_dropconnect = lambda V, pkeep: tf.nn.dropout(V, pkeep) * pkeep

    if doDropout:
        V4 = tf.nn.dropout(V4, pkeep)
    elif doDropConnect:
        V4 = tf_dropconnect(V4, pkeep)

    # V4[batch, FC4]
    b5 = tf.matmul(V4, W5) + T5
    O = tf.nn.softmax(b5)
    # O[batch, L5]

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

    # training, learning rate = eta (0.001275)
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

            acc_trn = 0
            loss_trn = 0
            for i in range(5):
                xTrainBatch = xTrainNorm[i * 10000:(i + 1) * 10000]
                tTrainBatch = tTrain[i * 10000:(i + 1) * 10000]

                # compute training values for visualization
                iacc_trn, iloss_trn = sess.run([accuracy, cross_entropy],
                                               feed_dict={X: xTrainBatch, Z: tTrainBatch, pkeep: 1.0})

                acc_trn += iacc_trn
                loss_trn += iloss_trn

            acc_trn /= 5.
            loss_trn /= 5.

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


