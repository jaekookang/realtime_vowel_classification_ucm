
import ipdb as pdb
import os
import shutil
import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import logging


def softmax_fun(x):
    '''Numerically stable softmax'''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def train_nn(train_data, test_data, vowels, training_epochs=1000, n_display=100, verbose=True, model_dir=None):
    '''ref: https://github.com/hunkim/DeepLearningZeroToAll'''
    if model_dir:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            os.mkdir(model_dir)
        else:
            os.mkdir(model_dir)

    # set logger
    if verbose:
        # print out in the notebook if verbose=True
        logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    if model_dir:
        handler = logging.FileHandler(os.path.join(model_dir, 'fwd_train.log'))
    else:
        handler = logging.FileHandler('fwd_train.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)

    # set tf graph
    tf.reset_default_graph()
    tf.set_random_seed(777)  # reproducibility

    # parameters
    learning_rate = 0.01
    keep_prob = 0.8
    logger.info(f'learning_rate: {learning_rate}')
    logger.info(f'training_epochs: {training_epochs}')
    logger.info(f'n_display: {n_display}')

    # dataset
    trainx, trainy = train_data
    testx, testy = test_data
    # trainy = np.reshape(trainy, (-1, 1))
    # testy = np.reshape(testy, (-1, 1))
    xdim = trainx.shape[1]
    # ydim = trainy.shape[1]
    ydim = len(vowels)

    logger.info('train_x: {}, train_y: {}'.format(trainx.shape, trainy.shape))
    logger.info('test_x: {}, test_y: {}\n'.format(testx.shape, testy.shape))

    X = tf.placeholder(tf.float32, [None, xdim], name='X')
    Y = tf.placeholder(tf.int64, [None], name='Y')
    Yonehot = tf.one_hot(Y, depth=len(vowels))

    hidden_output_size = 500
    final_output_size = ydim
    logger.info(f'hidden_output_size: {hidden_output_size}')
    logger.info(f'final_output_size: {final_output_size}\n')

    W1 = tf.get_variable("W1", shape=[xdim, hidden_output_size],
                         initializer=tf.contrib.layers.xavier_initializer(uniform=False))  # 5x300
    b1 = tf.Variable(tf.random_normal([hidden_output_size]))  # 300
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # Nx300

    W2 = tf.get_variable("W2", shape=[hidden_output_size, hidden_output_size],
                         initializer=tf.contrib.layers.xavier_initializer(uniform=False))  # 300x300
    b2 = tf.Variable(tf.random_normal([hidden_output_size]))  # 300
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)  # 300x300

    W3 = tf.get_variable("W3", shape=[hidden_output_size, hidden_output_size],
                         initializer=tf.contrib.layers.xavier_initializer(uniform=False))  # 300x300
    b3 = tf.Variable(tf.random_normal([hidden_output_size]))  # 300
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)  # 300x300

    W4 = tf.get_variable("W4", shape=[hidden_output_size, final_output_size],
                         initializer=tf.contrib.layers.xavier_initializer(uniform=False))  # 300x3
    b4 = tf.Variable(tf.random_normal([final_output_size]))  # 3
    logits = tf.matmul(L3, W4) + b4
    softmax = tf.nn.softmax(logits)
    correct_prediction = tf.equal(
        tf.round(tf.argmax(softmax, axis=1)), tf.round(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=Y, logits=logits)  # Cross Entropy Loss
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver(max_to_keep=30)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_cost = [0]  # pad 0 due to slope calculation
        total_test = [0]  # pad 0 due to slope calculation
        slope_avg = [0, 0, 0, 0]
        moving_avg = []
        # disp_threshold = 0.05
        disp_threshold = 3
        moving_avg += disp_threshold * [0]

        disp_cnt = 0
        for i in range(training_epochs):
            opt, train_c = sess.run([optimizer, cost], feed_dict={
                                    X: trainx, Y: trainy})
            if (i + 1) % n_display == 0:
                disp_cnt += 1

                params = dict(W1=W1.eval(), W2=W2.eval(), W3=W3.eval(),
                              b1=b1.eval(), b2=b2.eval(), b3=b3.eval())
                total_cost.append(np.mean(train_c))
                test_c, test_acc = sess.run(
                    [cost, accuracy], feed_dict={X: testx, Y: testy})
                total_test.append(np.mean(test_c))
                slope_avg.append(np.mean(test_c) - total_test[-2])
                logger.info(
                    '{:>4}/{} train: CE={:.4f}, test: CE={:.4f}, test_acc:{:.2f} slop={:>.5f}'.format(
                        i + 1, training_epochs,
                        np.mean(train_c), np.mean(test_c), np.mean(test_acc), slope_avg[-1]))

                if model_dir:
                    saver.save(sess, os.path.join(
                        model_dir, 'model_epoch={}'.format(i + 1)))

                if disp_cnt > disp_threshold:
                    rmse_over_test = np.sqrt(
                        np.mean(np.square(slope_avg[-3:])))
                    moving_avg.append(rmse_over_test)
#                     if moving_avg[-1] < threshold:
#                         break

        logging.info('Training finished')
        return total_cost[1:], total_test[1:], moving_avg, i + 1, params


def predict(model_dir, which_epoch, which_subj, indata, outdata=None):
    tf.reset_default_graph()

    print('model_dir:', model_dir)
    print('final_epoch:', which_epoch)
    print('which_subj:', which_subj)
    print('indata:', indata.shape)
    if outdata is not None:
        print('outdata:', outdata.shape)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(model_dir, 'model_epoch={}.meta'.format(which_epoch)))
        saver.restore(sess, os.path.join(
            model_dir, 'model_epoch={}'.format(which_epoch)))
        graph = tf.get_default_graph()

        # retrieve variables
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        pred = graph.get_tensor_by_name('add_2:0')
        if outdata is not None:
            cost = graph.get_tensor_by_name('cost:0')
            yhat, rmse = sess.run([pred, cost], feed_dict={
                                  X: indata, Y: outdata})
            return yhat, rmse
        else:
            yhat = sess.run(pred, feed_dict={X: indata})
            return yhat


def get_param(model_dir, which_epoch):
    tf.reset_default_graph()

    print('model_dir:', model_dir)
    print('which_epoch:', which_epoch)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(model_dir, 'model_epoch={}.meta'.format(which_epoch)))
        saver.restore(sess, os.path.join(
            model_dir, 'model_epoch={}'.format(which_epoch)))
        graph = tf.get_default_graph()

        # retrieve variables
        W1 = graph.get_tensor_by_name('W1:0')
        W2 = graph.get_tensor_by_name('W2:0')
        W3 = graph.get_tensor_by_name('W3:0')
        b1 = graph.get_tensor_by_name('Variable:0')
        b2 = graph.get_tensor_by_name('Variable_1:0')
        b3 = graph.get_tensor_by_name('Variable_2:0')

        params = dict(W1=W1.eval(), W2=W2.eval(), W3=W3.eval(),
                      b1=b1.eval(), b2=b2.eval(), b3=b3.eval())
        return params


def forward(X, params):
    '''
    This function requires ANN hyper-parameters and sigmoid function
    from outside.
    They were separated for calculating Jacobian matrix (numdifftools)
    '''
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    b1, b2, b3 = params['b1'], params['b2'], params['b3']
    #     L1 = relu(np.dot(X, W1) + b1) # Nx300
    #     L2 = relu(np.dot(L1, W2) + b2) # Nx300
    L1 = sigmoid(np.dot(X, W1) + b1)  # Nx300
    L2 = sigmoid(np.dot(L1, W2) + b2)  # Nx300
    return np.dot(L2, W3) + b3  # Nx3''''


def sigmoid(X):
    return 1 / (1 + np.exp(-X))
