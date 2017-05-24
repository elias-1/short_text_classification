#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: mt_dkgam.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017-04-02 16:11:46
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
tf.app.flags.DEFINE_string('log_dir', "mt_dkgam_logs", 'The log  dir')

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
tf.app.flags.DEFINE_integer("joint_steps", 600, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

tf.app.flags.DEFINE_float("num_classes", 14, "Number of classes to classify")
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


def linear(args, output_size, bias, bias_start=0.0, scope=None, reuse=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError('`args` must be specified')
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                'Linear is expecting 2D arguments: %s' % str(shapes))
        if not shape[1]:
            raise ValueError(
                'Linear expects shape[1] of arguments: %s' % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or 'Linear', reuse=reuse):
        matrix = tf.get_variable('Matrix', [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            'Bias', [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term


class Model:
    def __init__(self, distinctTagNum, numHidden):
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.words = tf.Variable(
            tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_size], -1.0,
                              1.0),
            name='words')

        self.entity_embedding_pad = tf.constant(
            0.0, shape=[1, numHidden * 2], name="entity_embedding_pad")

        self.entity_embedding = tf.Variable(
            tf.random_uniform([FLAGS.max_replace_entity_nums, numHidden * 2],
                              -1.0, 1.0),
            name="entity_embedding")

        self.entity_emb = tf.concat(
            [self.entity_embedding_pad, self.entity_embedding],
            0,
            name='entity_emb')

        with tf.variable_scope('Ner_output') as scope:
            self.W = tf.get_variable(
                shape=[numHidden * 2, distinctTagNum],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum], name="bias"))

        with tf.variable_scope('Attention') as scope:
            self.attend_W = tf.get_variable(
                "attend_W",
                shape=[1, 1, self.numHidden * 2, self.numHidden * 2],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

            self.attend_V = tf.get_variable(
                "attend_V",
                shape=[self.numHidden * 2],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

        with tf.variable_scope('Clfier_output') as scope:
            self.clfier_softmax_W = tf.get_variable(
                "clfier_W",
                shape=[numHidden * 2, FLAGS.num_classes],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

            self.clfier_softmax_b = tf.get_variable(
                "clfier_softmax_b",
                shape=[FLAGS.num_classes],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

        self.inp_w = tf.placeholder(
            tf.int32, shape=[None, FLAGS.max_sentence_len], name="input_words")

        self.entity_info = tf.placeholder(
            tf.int32,
            shape=[None, FLAGS.max_replace_entity_nums],
            name="entity_info")

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self,
                  wX,
                  model='ner',
                  entity_info=None,
                  rnn_reuse=None,
                  linear_resue=None,
                  trainMode=True):

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

        if model == 'ner':
            output = tf.reshape(output, [-1, self.numHidden * 2])
            matricized_unary_scores = tf.matmul(output, self.W) + self.b
            # matricized_unary_scores = tf.nn.log_softmax(matricized_unary_scores)
            unary_scores = tf.reshape(matricized_unary_scores, [
                -1, FLAGS.max_sentence_len, self.distinctTagNum
            ])

            return unary_scores, length
        elif model == 'clfier':
            entity_emb = tf.nn.embedding_lookup(self.entity_emb, entity_info)

            hidden = tf.reshape(
                output, [-1, FLAGS.max_sentence_len, 1, self.numHidden * 2])
            hidden_feature = tf.nn.conv2d(hidden, self.attend_W, [1, 1, 1, 1],
                                          "SAME")
            query = tf.reduce_sum(entity_emb, axis=1)
            y = linear(query, self.numHidden * 2, True, reuse=linear_resue)
            y = tf.reshape(y, [-1, 1, 1, self.numHidden * 2])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(self.attend_V * tf.tanh(hidden_feature + y),
                              [2, 3])
            a = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                tf.reshape(a, [-1, FLAGS.max_sentence_len, 1, 1]) * hidden,
                [1, 2])
            ds = tf.reshape(d, [-1, self.numHidden * 2])

            scores = tf.nn.xw_plus_b(ds, self.clfier_softmax_W,
                                     self.clfier_softmax_b)
            return scores, length
        else:
            raise ValueError('model must either be clfier or ner')

    def ner_loss(self, ner_wX, ner_Y):
        P, sequence_length = self.inference(ner_wX, model='ner')
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, ner_Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        regularization_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + regularization_loss * FLAGS.l2_reg_lambda

    def clfier_loss(self, clfier_wX, clfier_Y, entity_info):
        self.scores, _ = self.inference(
            clfier_wX, model='clfier', entity_info=entity_info, rnn_reuse=True)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.scores, labels=clfier_Y)
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        regularization_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        normed_embedding = tf.nn.l2_normalize(self.entity_emb, dim=1)
        similarity_matrix = tf.matmul(normed_embedding,
                                      tf.transpose(normed_embedding, [1, 0]))
        fro_norm = tf.reduce_sum(tf.nn.l2_loss(similarity_matrix))
        final_loss = loss + regularization_loss * FLAGS.l2_reg_lambda + fro_norm * FLAGS.matrix_norm
        return final_loss

    def test_unary_score(self):
        P, sequence_length = self.inference(
            self.inp_w, model='ner', rnn_reuse=True, trainMode=False)
        return P, sequence_length

    def test_clfier_score(self):
        scores, _ = self.inference(
            self.inp_w,
            model='clfier',
            entity_info=self.entity_info,
            rnn_reuse=True,
            linear_resue=True,
            trainMode=False)
        return scores


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


def train(total_loss, var_list=None):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        total_loss, var_list=var_list)


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
    entity_infos = []
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
            entity_infos.append(viterbi_sequence)
            result_tag_list.append([entity_tag[i] for i in viterbi_sequence])
            ref_tag_list.append([entity_tag[i] for i in y_])
            word_list.append([vocab[i] for i in word_x_])

    tagging_eval_result = conlleval(result_tag_list, ref_tag_list, word_list,
                                    FLAGS.taging_out_file)
    print("precision: %.2f, recall: %.2f, f1-score: %.2f" %
          (tagging_eval_result['p'], tagging_eval_result['r'],
           tagging_eval_result['f1']))
    return entity_infos


def clfier_test_evaluate(sess, test_clfier_score, inp_w, entity_info,
                         clfier_twX, clfier_tY, tentity_info):
    batchSize = FLAGS.batch_size
    totalLen = clfier_twX.shape[0]
    numBatch = int((totalLen - 1) / batchSize) + 1
    correct_clfier_labels = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = clfier_tY[i * batchSize:endOff]
        feed_dict = {
            inp_w: clfier_twX[i * batchSize:endOff],
            entity_info: tentity_info[i * batchSize:endOff]
        }
        clfier_score_val = sess.run([test_clfier_score], feed_dict)
        predictions = np.argmax(clfier_score_val[0], 1)
        correct_clfier_labels += np.sum(np.equal(predictions, y))

    accuracy = 100.0 * correct_clfier_labels / float(totalLen)
    print("Clfier Accuracy: %.3f%%" % accuracy)


def decode_entity_location(entity_info):
    entity_location = []
    types_id = []
    loc = 0
    while loc < len(entity_info):
        # tag:PAD O
        if entity_info[loc] < 2:
            loc += 1
            continue
        # tag: B
        elif (entity_info[loc] - 2) % 2 == 0:
            types_id.append((entity_info[loc] - 2) / 4)
            length = 1
            while loc + length < len(
                    entity_info) and entity_info[loc + length] == (
                        entity_info[loc] + 1):
                length += 1
            entity_location.append([loc, loc + length - 1])
            loc += length
            continue
        else:
            # print(
            #     'the entity info is not discordant with the ios tagging scheme')
            loc += 1
    types_id = map(lambda x: int(x) + 1, types_id)
    return entity_location, types_id


def entity_encode(entity_infos):
    tentity_info = []
    for i in range(len(entity_infos)):
        entity_location, types_id = decode_entity_location(entity_infos[i])
        nl = len(types_id)

        for i in range(nl, FLAGS.max_replace_entity_nums):
            types_id.append('0')
        tentity_info.append(types_id[:FLAGS.max_replace_entity_nums])

    return np.array(tentity_info)


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
        ner_var_list = [
            v for v in tf.global_variables()
            if 'Attention' not in v.name and 'Clfier_output' not in v.name and
            'Linear' not in v.name
        ]

        ner_train_op = train(ner_total_loss, var_list=ner_var_list)
        ner_test_unary_score, ner_test_sequence_length = model.test_unary_score(
        )

        clfier_total_loss = model.clfier_loss(clfier_wX, clfier_Y, entity_info)
        clfier_var_list = [
            v for v in tf.global_variables()
            if 'Ner_output' not in v.name and 'transitions' not in v.name and
            'Adam' not in v.name
        ]

        clfier_train_op = train(clfier_total_loss, var_list=clfier_var_list)
        test_clfier_score = model.test_clfier_score()

        ner_seperate_list = [
            v for v in tf.global_variables()
            if 'Ner_output' in v.name or 'transition' in v.name
        ]
        ner_seperate_op = train(ner_total_loss, var_list=ner_seperate_list)

        clfier_seperate_list = [
            v for v in tf.global_variables()
            if 'Attention' in v.name or 'Clfier_output' in v.name or
            'Linear' in v.name
        ]
        clfier_seperate_op = train(
            ner_total_loss, var_list=clfier_seperate_list)

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
                    if step < FLAGS.joint_steps:
                        _, trainsMatrix = sess.run(
                            [ner_train_op, model.transition_params])
                    else:
                        _, trainsMatrix = sess.run(
                            [ner_seperate_op, model.transition_params])
                    # for debugging and learning purposes, see how the loss gets decremented thru training steps
                    if (step + 1) % 10 == 0:
                        print(
                            "[%d] NER loss: [%r]    Classification loss: [%r]"
                            % (step + 1, sess.run(ner_total_loss),
                               sess.run(clfier_total_loss)))
                    if (step + 1) % 20 == 0:
                        entity_infos = ner_test_evaluate(
                            sess, ner_test_unary_score,
                            ner_test_sequence_length, trainsMatrix,
                            model.inp_w, ner_twX, ner_tY)
                        tentity_info = entity_encode(entity_infos)
                        clfier_test_evaluate(sess, test_clfier_score,
                                             model.inp_w, model.entity_info,
                                             clfier_twX, clfier_tY,
                                             tentity_info)
                    if step < FLAGS.joint_steps:
                        _ = sess.run([clfier_train_op])
                    else:
                        _ = sess.run([clfier_seperate_op])

                except KeyboardInterrupt as e:
                    sv.saver.save(
                        sess, FLAGS.log_dir + '/model', global_step=(step + 1))
                    raise e
            sv.saver.save(sess, FLAGS.log_dir + '/finnal-model')


if __name__ == '__main__':
    tf.app.run()
