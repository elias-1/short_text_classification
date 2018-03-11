#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: signal_recognition.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2018-03-11 09:36:14
"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from func_utils import NUM_TAGS, MAX_SEQ_LEN, SIGNAL_WIDTH

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', "train_data.csv",
                           'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', "test_data.csv",
                           'Test data dir')
tf.app.flags.DEFINE_string('log_dir', "ner_logs", 'The log  dir')

tf.app.flags.DEFINE_integer("max_sentence_len", MAX_SEQ_LEN,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("signal_width", SIGNAL_WIDTH,
                            "signal width per test")
tf.app.flags.DEFINE_integer("num_hidden", 5, "hidden unit number")
tf.app.flags.DEFINE_integer("num_tags", NUM_TAGS, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 2000, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.7,
                          'Dropout keep probability (default: 0.7)')

tf.flags.DEFINE_float('l2_reg_lambda', 0,
                      'L2 regularization lambda (default: 0.0)')


class Model:
    def __init__(self, distinctTagNum, numHidden):
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden

        with tf.variable_scope('Ner_output') as scope:
            self.W = tf.get_variable(
                shape=[numHidden * 2, distinctTagNum],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum], name="bias"))

        self.signal_emd = tf.placeholder(
            tf.float32, shape=[None, FLAGS.max_sentence_len, FLAGS.signal_width], name="signal_emd")

    def length(self, data):
        seq_data = tf.reduce_sum(data, reduction_indices=2)
        used = tf.sign(tf.abs(seq_data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, ner_X, rnn_reuse=None, trainMode=True):

        word_vectors = tf.reshape(ner_X, [-1, FLAGS.max_sentence_len, FLAGS.signal_width])
        length = self.length(word_vectors)
        length_64 = tf.cast(length, tf.int64)

        # if trainMode:
        #  word_vectors = tf.nn.dropout(word_vectors, FLAGS.dropout_keep_prob)
        with tf.variable_scope("rnn_fwbw", reuse=rnn_reuse) as scope:
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.numHidden),
                word_vectors,
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.numHidden),
                inputs=tf.reverse_sequence(word_vectors, length_64, seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backword")

        backward_output = tf.reverse_sequence(
            backward_output_, length_64, seq_dim=1)

        output = tf.concat([forward_output, backward_output], 2)
        if trainMode:
            output = tf.nn.dropout(output, FLAGS.dropout_keep_prob)

        output = tf.reshape(output, [-1, self.numHidden * 2])
        matricized_unary_scores = tf.matmul(output, self.W) + self.b
        # matricized_unary_scores = tf.nn.log_softmax(matricized_unary_scores)
        unary_scores = tf.reshape(matricized_unary_scores, [
            -1, FLAGS.max_sentence_len, self.distinctTagNum
        ])

        return unary_scores, length

    def ner_loss(self, ner_X, ner_Y):
        P, sequence_length = self.inference(ner_X)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, ner_Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        regularization_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + regularization_loss * FLAGS.l2_reg_lambda

    def test_unary_score(self):
        P, sequence_length = self.inference(
            self.signal_emd, rnn_reuse=True, trainMode=False)
        return P, sequence_length


def read_csv(batch_size, file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(
        value,
        field_delim=' ',
        record_defaults=[[0.0]]*FLAGS.max_sentence_len * FLAGS.signal_width
                        + [[0]]*FLAGS.max_sentence_len)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(
        decoded,
        batch_size=batch_size,
        capacity=batch_size * 4,
        min_after_dequeue=batch_size)


def inputs(path):
    whole = read_csv(FLAGS.batch_size, path)
    ner_features = tf.transpose(
        tf.stack(whole[0:FLAGS.max_sentence_len*FLAGS.signal_width]))

    ner_label = tf.transpose(
        tf.stack(whole[FLAGS.max_sentence_len*FLAGS.signal_width:FLAGS.max_sentence_len*(FLAGS.signal_width+1)]))

    return ner_features, ner_label


def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)


def ner_test_evaluate(sess, unary_score, test_sequence_length, transMatrix,
                      inp_ner_w, ner_X, ner_Y):
    batchSize = FLAGS.batch_size
    totalLen = ner_X.shape[0]
    numBatch = int((totalLen - 1) / batchSize) + 1
    correct_labels = 0
    total_labels = 0

    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = ner_Y[i * batchSize:endOff]
        feed_dict = {inp_ner_w: ner_X[i * batchSize:endOff]}
        unary_score_val, test_seq_len_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)
        for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_seq_len_val):
            # print("seg len:%d" % (test_seq_len_val))
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, transMatrix)
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += sequence_length_
    accuracy = 100.0 * correct_labels / float(total_labels)
    print("Accuracy: %.3f%%" % accuracy)


def load_test_data(path, max_sentence_len, signal_width):
    ner_x = []
    ner_y = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        assert (len(ss) == (max_sentence_len * (signal_width+1))), \
            "[line:%d]len ss:%d,origin len:%d\n%s" % (ln, len(ss), len(line), line)
        lner_x = []
        lner_y = []
        for i in range(max_sentence_len*signal_width):
            lner_x.append(float(ss[i]))

        for i in range(max_sentence_len):
            lner_y.append(int(ss[max_sentence_len*signal_width + i]))

        ner_x.append(lner_x)
        ner_y.append(lner_y)
    fp.close()
    return np.array(ner_x).reshape(-1, max_sentence_len, signal_width), np.array(ner_y)


def main(unused_argv):
    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.num_tags, FLAGS.num_hidden)
        print("train data path:", os.path.realpath(FLAGS.train_data_path))
        ner_X, ner_Y = inputs(
            FLAGS.train_data_path)
        ner_tX, ner_tY = load_test_data(
            FLAGS.test_data_path, FLAGS.max_sentence_len,
            FLAGS.signal_width)

        ner_total_loss = model.ner_loss(ner_X, ner_Y)
        ner_train_op = train(ner_total_loss)
        ner_test_unary_score, ner_test_sequence_length = model.test_unary_score(
        )

        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        with sv.managed_session(
                master='',
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    _, trainsMatrix = sess.run(
                        [ner_train_op, model.transition_params])
                    # for debugging and learning purposes, see how the loss gets decremented thru training steps
                    if (step + 1) % 10 == 0:
                        print("[%d] loss: [%r]" % (step + 1,
                                                   sess.run(ner_total_loss)))
                    if (step + 1) % 20 == 0:
                        ner_test_evaluate(sess, ner_test_unary_score,
                                          ner_test_sequence_length,
                                          trainsMatrix, model.signal_emd, ner_tX,
                                          ner_tY)

                except KeyboardInterrupt as e:
                    sv.saver.save(
                        sess, FLAGS.log_dir + '/model', global_step=(step + 1))
                    raise e
            sv.saver.save(sess, FLAGS.log_dir + '/finnal-model')


if __name__ == '__main__':
    tf.app.run()
