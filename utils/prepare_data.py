#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: prepare_data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/21 11:39

1. create vocabulary 
python prepare_data.py -create_vocabulary -entity_intent ../data/entity_intent_types.txt \
   -vocabulary_filename ../data/vocab.txt -train_dir ../data/train

4. cnn/bilstm/rcnn
python prepare_data.py -entity_intent ../data/entity_intent_types.txt -vocabulary_filename \
 ../data/vocab.txt -train_dir ../data/train -test_dir ../data/test -task_type 3 -task_name normal \
 -max_sentence_len 20 -max_replace_entity_nums 5

5. cnn/bilstm/rcnn entity words replacing
python prepare_data.py -entity_intent ../data/entity_intent_types.txt -vocabulary_filename \
 ../data/vocab.txt -train_dir ../data/train -test_dir ../data/test -task_type 4 -task_name replacing \
 -max_sentence_len 20 -max_replace_entity_nums 5


"""

import argparse
import os

UNK = '<UNK>'
PAD = '<PAD>'


def arg_parser():
    parser = argparse.ArgumentParser(
        prog='prepare_data',
        formatter_class=argparse.RawTextHelpFormatter,
        description='Prepare data for input of deep model')

    parser.add_argument(
        '-create_vocabulary',
        dest='create_vocabulary',
        action='store_true',
        help='Whether to create vocabulary')

    parser.add_argument(
        '-entity_intent',
        dest='entity_intent_collected',
        default='../data/entity_intent_types.txt',
        type=str,
        help='entity and intent collected for stats')

    parser.add_argument(
        '-vocabulary_filename',
        dest='vocabulary_filename',
        default='../data/vocab.txt',
        type=str,
        help='filename of vocabulary.')

    parser.add_argument(
        '-train_dir',
        dest='train_dir',
        default='../data/train',
        type=str,
        help='dir for training data')

    parser.add_argument(
        '-test_dir',
        dest='test_dir',
        default='../data/test',
        type=str,
        help='dir for test data')

    parser.add_argument(
        '-task_type',
        dest='task_type',
        type=int,
        help=
        '1 for dkgam; 2 for mt-dkgam; 3 for cnn/bi-lstm/rcnn; 4 for cnn/bi-lstm common;'
    )

    parser.add_argument(
        '-task_name',
        dest='task_name',
        type=str,
        help='used for data filename of training/test')

    parser.add_argument(
        '-max_sentence_len',
        dest='max_sentence_len',
        default=20,
        type=int,
        help='used for data filename of training/test')

    parser.add_argument(
        '-max_replace_entity_nums',
        dest='max_replace_entity_nums',
        default=5,
        type=int,
        help='max replace entity nums for entity words replacing')

    return parser


def create_vocabulary(vocabulary_filename, train_dir):
    vocab = {UNK: 0}
    with open(os.path.join(train_dir, 'train.seq.in'), 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            for token in tokens:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1

    with open(vocabulary_filename, 'w') as f:
        for token in vocab.keys():
            f.write(token + '\t' + str(vocab[token]) + '\n')


def entity_collected(entity_intent_collected, train_dir):
    entity_types = []
    intent_types = []
    with open(os.path.join(train_dir, 'train.seq.out'), 'r') as f:
        for line in f.readlines():
            slots = line.strip().split()
            for slot in slots:
                if slot == 'O':
                    continue
                if '-' in slot:
                    slot = slot.split('-')[1]
                if slot not in entity_types:
                    entity_types.append(slot)

    with open(os.path.join(train_dir, 'train.label'), 'r') as f:
        for line in f.readlines():
            intent_type = line.strip()
            if intent_type not in intent_types:
                intent_types.append(intent_type)

    with open(entity_intent_collected, 'w') as f:
        f.write(' '.join(entity_types) + '\n')
        f.write(' '.join(intent_types) + '\n')


def data_preprocess(data_dir, entity_types, intent_types, vocab,
                    max_sentence_len, max_replace_entity_nums):
    data = []
    data_common = []
    slot_labels = []
    intent_labels = []
    entity_labels = []
    train_or_test = os.path.basename(data_dir)
    data_fp = open(os.path.join(data_dir, train_or_test + '.seq.in'), 'r')
    intent_fp = open(os.path.join(data_dir, train_or_test + '.label'), 'r')
    slot_fp = open(os.path.join(data_dir, train_or_test + '.seq.out'), 'r')
    vocab_index = {vocab[i]: str(i) for i in range(len(vocab))}
    while True:
        tokens = data_fp.readline().strip().split()

        if not tokens:
            break

        try:
            intent_x = str(intent_types.index(intent_fp.readline().strip()))
        except:
            continue
        intent_labels.append(intent_x)

        sample_x = []
        sample_len = len(tokens)
        for token in tokens:
            try:
                sample_x.append(vocab_index[token])
            except:
                sample_x.append(vocab_index[UNK])
        for i in range(sample_len, max_sentence_len):
            sample_x.append('0')
        data.append(sample_x[:max_sentence_len])

        slots = slot_fp.readline().strip().split()
        slot_type_index = [PAD, 'O']
        for entity in entity_types:
            slot_type_index.extend(['B-' + entity, 'I-' + entity])
        slots_x = []
        for slot in slots:
            try:
                slots_x.append(str(slot_type_index.index(slot)))
            except:
                slots_x.append(str(slot_type_index.index('O')))
        for i in range(sample_len, max_sentence_len):
            slots_x.append('0')
        slot_labels.append(slots_x[:max_sentence_len])

        entity_x = []
        i = 0
        while i < len(slots):
            if slots[i] == 'O':
                i += 1
                continue
            elif 'B-' in slots[i]:
                try:
                    entity_x.append(
                        str(entity_types.index(slots[i].split('-')[1]) + 1))
                except:
                    pass
                j = 1
                while i + j < len(slots) and 'I-' in slots[i + j]:
                    j += 1
                i += j
                continue
            else:
                raise ValueError(
                    'slots in train.seq.in/test.seq.in may have error')
        for i in range(len(entity_x), max_replace_entity_nums):
            entity_x.append('0')
        entity_labels.append(entity_x[:max_replace_entity_nums])

        sample_common_x = []
        i = 0
        while i < len(slots):
            if slots[i] == 'O':
                try:
                    sample_common_x.append(vocab_index[tokens[i]])
                except:
                    sample_common_x.append(vocab_index[UNK])
                i += 1
                continue
            elif 'B-' in slots[i]:
                try:
                    sample_common_x.append(
                        str(
                            entity_types.index(slots[i].split('-')[1]) +
                            len(vocab)))
                except:
                    pass
                j = 1
                while i + j < len(slots) and 'I-' in slots[i + j]:
                    j += 1
                i += j
                continue
            else:
                raise ValueError(
                    'slots in train.seq.in/test.seq.in may have error')
        for i in range(len(sample_common_x), max_sentence_len):
            sample_common_x.append('0')
        data_common.append(sample_common_x[:max_sentence_len])

    data_fp.close()
    intent_fp.close()
    slot_fp.close()

    return data, data_common, slot_labels, intent_labels, entity_labels


def prepare_data_for_dkgam(train, test, train_dir, test_dir, task_name):
    def make_data_set(data, filename):
        data_set = []
        for sample_x, entity_x, intent_x in zip(data[0], data[1], data[2]):
            data_set.append(sample_x + [
                intent_x,
            ] + entity_x)

        with open(filename, 'w') as f:
            for line in data_set:
                f.write(' '.join(line) + '\n')

    make_data_set(train, os.path.join(train_dir, task_name + '_train.txt'))
    make_data_set(test, os.path.join(test_dir, task_name + '_test.txt'))


def prepare_data_for_mt_dkgam(train, test, train_dir, test_dir, task_name):
    def make_data_set(data, filename):
        data_set = []
        for sample_x, slot_x, intent_x, entity_x in zip(
                data[0], data[1], data[2], data[3]):
            data_set.append(sample_x + slot_x + [
                intent_x,
            ] + entity_x)

        with open(filename, 'w') as f:
            for line in data_set:
                f.write(' '.join(line) + '\n')

    make_data_set(train, os.path.join(train_dir, task_name + '_train.txt'))
    make_data_set(test, os.path.join(test_dir, task_name + '_test.txt'))


def prepare_data_for_normal(train, test, train_dir, test_dir, task_name):
    def make_data_set(data, filename):
        data_set = []
        for sample_x, intent_x in zip(data[0], data[1]):
            data_set.append(sample_x + [
                intent_x,
            ])

        with open(filename, 'w') as f:
            for line in data_set:
                f.write(' '.join(line) + '\n')

    make_data_set(train, os.path.join(train_dir, task_name + '_train.txt'))
    make_data_set(test, os.path.join(test_dir, task_name + '_test.txt'))


def output_task_data(train_dir, test_dir, entity_types, intent_types, vocab,
                     task_type, task_name, max_sentence_len,
                     max_replace_entity_nums):
    train_data, train_common_data, train_slot_labels, train_intent_labels, train_entity_labels = data_preprocess(
        data_dir=train_dir,
        entity_types=entity_types,
        intent_types=intent_types,
        vocab=vocab,
        max_sentence_len=max_sentence_len,
        max_replace_entity_nums=max_replace_entity_nums)

    test_data, test_common_data, test_slot_labels, test_intent_labels, test_entitys_labels = data_preprocess(
        data_dir=test_dir,
        entity_types=entity_types,
        intent_types=intent_types,
        vocab=vocab,
        max_sentence_len=max_sentence_len,
        max_replace_entity_nums=max_replace_entity_nums)

    if task_type == 1:
        train = [train_data, train_entity_labels, train_intent_labels]
        test = [test_data, test_entitys_labels, test_intent_labels]
        prepare_data_for_dkgam(train, test, train_dir, test_dir, task_name)
    elif task_type == 2:
        train = [
            train_data, train_slot_labels, train_intent_labels,
            train_entity_labels
        ]
        test = [
            test_data, train_slot_labels, test_intent_labels,
            train_entity_labels
        ]
        prepare_data_for_mt_dkgam(train, test, train_dir, test_dir, task_name)
    elif task_type == 3:
        train = [train_data, train_intent_labels]
        test = [test_data, test_intent_labels]
        prepare_data_for_normal(train, test, train_dir, test_dir, task_name)
    elif task_type == 4:
        train = [train_common_data, train_intent_labels]
        test = [test_common_data, test_intent_labels]
        prepare_data_for_normal(train, test, train_dir, test_dir, task_name)
    else:
        raise ValueError('--task_type must be in [1,2,3,4]')


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    if args.create_vocabulary and args.task_type:
        raise ValueError(
            '--create_vocabulary and --task_type can not occur simultaneously, you should firstly'
            'create_vocabulary using --create_vocabulary, and then specify the --task_type '
            'to get wanted data')

    if args.task_type and not args.task_name:
        raise ValueError(
            '--task_name must be given which will be used for output filename')

    if args.create_vocabulary:
        create_vocabulary(args.vocabulary_filename, args.train_dir)
        entity_collected(args.entity_intent_collected, args.train_dir)
        exit(0)

    vocab = [
        PAD,
        UNK,
    ]
    with open(args.vocabulary_filename, 'r') as f:
        for line in f.readlines():
            vocab.append(line.strip().split('\t')[0])

    entity_types = []
    intent_types = []
    with open(args.entity_intent_collected, 'r') as f:
        line = f.readline()
        entity_types += line.strip().split()
        line = f.readline()
        intent_types = line.strip().split()

    output_task_data(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        entity_types=entity_types,
        intent_types=intent_types,
        vocab=vocab,
        task_type=args.task_type,
        task_name=args.task_name,
        max_sentence_len=args.max_sentence_len,
        max_replace_entity_nums=args.max_replace_entity_nums)
