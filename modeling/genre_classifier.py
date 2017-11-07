# Import Libraries
import numpy as np
import tensorflow as tf
import scipy
import time

import data_loader as DataLoader

get_class_from_genre = {
    'Electronic': 0,
    'Experimental': 1,
    'Folk': 2,
    'Hip-Hop': 3,
    'Instrumental': 4,
    'International': 5,
    'Pop': 6,
    'Rock': 7
}


class GenreClassifier(object):

    def __init__(self, num_epochs=30, batch_size=40, num_iterations=800, learning_rate=5e-4, input_dimension=(120, 300, 1), num_classes=8, log_step=200):
        self.start_time = time.time()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.log_step = log_step

        self.build_model()

    def set_input(self):
        self.X = tf.placeholder(tf.float32, [None, *self.input_dimension])
        self.y = tf.placeholder(tf.int64, [None, self.num_classes])

        self.is_train = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

    def model(self):

        _ = tf.layers.conv2d(
            inputs=self.X,
            filters=32,
            kernel_size=[5, 5],
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu,
            data_format='channels_last',
            name='conv1'
        )

        _ = tf.layers.max_pooling2d(
            inputs=_,
            pool_size=[2, 1],
            strides=[2, 1],
            padding='valid',
            data_format='channels_last',
            name='max_pool1'
        )

        _ = tf.layers.conv2d(
            inputs=_,
            filters=32,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu,
            data_format='channels_last',
            name='conv2'
        )

        _ = tf.layers.max_pooling2d(
            inputs=_,
            pool_size=[2, 1],
            strides=[2, 1],
            padding='valid',
            data_format='channels_last',
            name='max_pool2'
        )

        _ = tf.layers.conv2d(
            inputs=_,
            filters=4,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu,
            data_format='channels_last',
            name='conv3'
        )

        _ = tf.transpose(_, perm=[0, 2, 1, 3])
        _ = tf.reshape(_, [-1, 300, 120])

        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(120)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(120)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])

        _, state = tf.nn.dynamic_rnn(lstm_cells, _, dtype=tf.float32)

        _ = _[:, -1, :]

        _ = tf.layers.dense(
            inputs=_,
            units=60,
            activation=tf.nn.relu,
        )

        _ = tf.layers.dense(
            inputs=_,
            units=self.num_classes
        )

        return _


    def loss_function(self, labels, logits):
        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    def build_optimizer(self):
        global_step = tf.Variable(0, trainable=False)
        exp_dec = tf.train.exponential_decay(
            self.learning_rate, global_step, 500, 0.96, staircase=True)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=exp_dec).minimize(self.loss_op)

    def build_model(self):
        # Define input variables
        self.set_input()

        # # Convert y to one-hot vector
        # labels = tf.one_hot(self.Y, 10)

        # Build a model and get logits
        logits = self.model()

        # Compute loss
        self.loss_function(self.y, logits)

        # Build optimizer
        self.build_optimizer()

        # Compute accuracy
        predicted_labels = tf.argmax(logits, axis=1)
        true_labels = tf.argmax(self.y, axis=1)
        correct = tf.equal(predicted_labels, true_labels)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    def train(self, sess, X_train, X_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        accuracies = []

        for epoch in range(self.num_epochs):
            for iteration in range(self.num_iterations):
                # Create input for batch creation
                X_files, y_labels = list(), list()
                for genre, files in X_train.items():
                    # Pop first element
                    file = files.pop(0)
                    X_files.append(file)
                    y_labels.append(get_class_from_genre[genre])

                    # Add back to queue
                    files.append(file)

                # For each iteration, get batches
                batches = DataLoader.get_batches(
                    paths=X_files, labels=y_labels, is_train=True, num_classes=self.num_classes)

                # Run each step
                for batch in batches:
                    X_ = batch['X']
                    X_ = np.reshape(X_, [X_.shape[0], X_.shape[1], X_.shape[2], 1])
                    y_ = batch['y']

                    feed_dict = {self.X: X_, self.y: y_,
                                 self.is_train: True, self.keep_prob: 0.7}
                    fetches = [self.train_op,
                               self.loss_op, self.accuracy_op]

                    _, loss, accuracy = sess.run(
                        fetches, feed_dict=feed_dict)

                    losses.append(loss)
                    accuracies.append(accuracy)

                    if step % self.log_step == 0:
                        print('step (%d): loss = %.3f, accuracy = %.3f, time elapsed = %.3f minutes' % (
                            step, loss, accuracy, ((time.time() - self.start_time)/60)))
                    step += 1

            # Evaluate validation set for each epoch
            print('Validation for epoch %d' % epoch)
            val_accuracy = self.evaluate(sess, X_val)
            print('>> EPOCH %d: Validation Accuracy = %.4f' %
                  (epoch, val_accuracy))

    def evaluate(self, sess, X_eval):
        # Create input for batch creation
        X_files, y_labels = list(), list()
        for genre, files in X_eval.items():
            X_files.extend(files)
            y_labels.extend([get_class_from_genre[genre]] * len(files))

        # For each iteration, get batches
        batches = DataLoader.get_batches(
            paths=X_files, labels=y_labels, is_train=False, num_classes=self.num_classes)

        eval_accuracy = 0.0
        eval_iter = 0.0
        for batch in batches:
            X_ = batch['X']
            X_ = np.reshape(X_, [X_.shape[0], X_.shape[1], X_.shape[2], 1])
            y_ = batch['y']

            feed_dict = {self.X: X_, self.y: y_, self.is_train: False}
            accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
            eval_accuracy += accuracy
            eval_iter += 1

        return eval_accuracy / eval_iter
