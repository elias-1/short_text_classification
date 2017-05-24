#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: train_ner.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017-04-02 16:11:46
Note: This file is just used to verify the performance of ner, so the input is the same as mt_dkgam.py.
"""

from __future__ import absolute_import, division, print_function

import os
import stat
import subprocess

import numpy as np
import tensorflow as tf
from func_utils import load_data_mt_dkgam

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', "data/train/mt_dkgam_train.txt",
                           'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', "data/test/mt_dkgam_test.txt",
                           'Test data dir')
tf.app.flags.DEFINE_string('log_dir', "ner_logs", 'The log  dir')

tf.app.flags.DEFINE_string("vocab_size", 934, "vocabulary size")
tf.app.flags.DEFINE_integer("max_sentence_len", 20,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("max_replace_entity_nums", 5,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 64, "embedding size")
tf.app.flags.DEFINE_integer("num_tags", 2 + 38 * 2, "num ner tags")
tf.app.flags.DEFINE_integer("num_hidden", 50, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 2000, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.7,
                          'Dropout keep probability (default: 0.7)')

tf.flags.DEFINE_float('l2_reg_lambda', 0,
                      'L2 regularization lambda (default: 0.0)')

tf.flags.DEFINE_float('matrix_norm', 0.01, 'frobieums norm (default: 0.01)')

tf.app.flags.DEFINE_string('vocabulary_filename', "data/vocab.txt",
                           'vocabulary file name')
tf.app.flags.DEFINE_string('entity_type_filename',
                           "data/entity_intent_types.txt",
                           'entity_type file name')
tf.app.flags.DEFINE_string('taging_out_file', "data/taging_result.txt",
                           'taging_result for conlleval.pl')


class Model:
    def __init__(self, distinctTagNum, numHidden):
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.words = tf.Variable(
            tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_size], -1.0,
                              1.0),
            name='words')

        with tf.variable_scope('Ner_output') as scope:
            self.W = tf.get_variable(
                shape=[numHidden * 2, distinctTagNum],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum], name="bias"))

        self.inp_w = tf.placeholder(
            tf.int32, shape=[None, FLAGS.max_sentence_len], name="input_words")

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, wX, rnn_reuse=None, trainMode=True):

        word_vectors = tf.nn.embedding_lookup(self.words, wX)
        length = self.length(wX)
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

    def ner_loss(self, ner_wX, ner_Y):
        P, sequence_length = self.inference(ner_wX)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, ner_Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        regularization_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + regularization_loss * FLAGS.l2_reg_lambda

    def test_unary_score(self):
        P, sequence_length = self.inference(
            self.inp_w, rnn_reuse=True, trainMode=False)
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
        record_defaults=[
            [0]
            for i in range(
                FLAGS.max_sentence_len * 2 + 1 + FLAGS.max_replace_entity_nums)
        ])

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(
        decoded,
        batch_size=batch_size,
        capacity=batch_size * 4,
        min_after_dequeue=batch_size)


def inputs(path):
    whole = read_csv(FLAGS.batch_size, path)
    ner_train_len = FLAGS.max_sentence_len * 2
    ner_features = clfier_features = tf.transpose(
        tf.stack(whole[0:FLAGS.max_sentence_len]))

    ner_label = tf.transpose(
        tf.stack(whole[FLAGS.max_sentence_len:2 * FLAGS.max_sentence_len]))

    clfier_label = tf.transpose(
        tf.concat(whole[ner_train_len:ner_train_len + 1], 0))
    entity_info = tf.transpose(tf.stack(whole[ner_train_len + 1:]))
    return ner_features, ner_label, clfier_features, clfier_label, entity_info


def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)


# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1])  # remove the ending \n on last line
    f.close()

    return get_perf(filename)


def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(
        ["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def prepare_vocab_and_tag():
    vocab = []
    with open(FLAGS.vocabulary_filename, 'r') as f:
        for line in f.readlines():
            vocab.append(line.strip().split('\t')[0])

    entity_tag = ['<PAD>', '<UNK>']
    with open(FLAGS.entity_type_filename, 'r') as f:
        line = f.readline()
        entity_types = line.strip().split()
        for entity_type in entity_types:
            entity_tag.append('B-' + entity_type)
            entity_tag.append('I-' + entity_type)

    return vocab, entity_tag


def ner_test_evaluate(sess, unary_score, test_sequence_length, transMatrix,
                      inp_ner_w, ner_wX, ner_Y):
    batchSize = FLAGS.batch_size
    totalLen = ner_wX.shape[0]
    numBatch = int((ner_wX.shape[0] - 1) / batchSize) + 1
    vocab, entity_tag = prepare_vocab_and_tag()
    result_tag_list = []
    ref_tag_list = []
    word_list = []
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = ner_Y[i * batchSize:endOff]
        feed_dict = {inp_ner_w: ner_wX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)
        for word_x_, tf_unary_scores_, y_, sequence_length_ in zip(
                ner_wX[i * batchSize:endOff], unary_score_val, y,
                test_sequence_length_val):
            # print("seg len:%d" % (sequence_length_))
            word_x_ = word_x_[:sequence_length_]
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, transMatrix)
            result_tag_list.append([entity_tag[i] for i in viterbi_sequence])
            ref_tag_list.append([entity_tag[i] for i in y_])
            word_list.append([vocab[i] for i in word_x_])

    tagging_eval_result = conlleval(result_tag_list, ref_tag_list, word_list,
                                    FLAGS.taging_out_file)
    print("precision: %.2f, recall: %.2f, f1-score: %.2f" %
          (tagging_eval_result['p'], tagging_eval_result['r'],
           tagging_eval_result['f1']))


def main(unused_argv):
    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.num_tags, FLAGS.num_hidden)
        print("train data path:", os.path.realpath(FLAGS.train_data_path))
        ner_wX, ner_Y, clfier_wX, clfier_Y, entity_info = inputs(
            FLAGS.train_data_path)
        ner_twX, ner_tY, clfier_twX, clfier_tY, _ = load_data_mt_dkgam(
            FLAGS.test_data_path, FLAGS.max_sentence_len,
            FLAGS.max_replace_entity_nums)

        ner_total_loss = model.ner_loss(ner_wX, ner_Y)
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
                                          trainsMatrix, model.inp_w, ner_twX,
                                          ner_tY)

                except KeyboardInterrupt as e:
                    sv.saver.save(
                        sess, FLAGS.log_dir + '/model', global_step=(step + 1))
                    raise e
            sv.saver.save(sess, FLAGS.log_dir + '/finnal-model')


if __name__ == '__main__':
    tf.app.run()
