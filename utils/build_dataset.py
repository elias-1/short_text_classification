#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: build_dataset.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/15 21:01

"""

import os
import random
import re
import sys

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
SPLIT_RATIO = 0.8


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def naive_tokenizer(sentence):
    """Naive tokenizer: split the sentence by space into a list of tokens."""
    return sentence.split()


def get_data_stats(data_stats_file):
    intent_stats = {}
    slot_stats = {}
    intent_over = 0
    with open(data_stats_file, 'r') as f:
        for line in f.readlines():
            line_columns = line.strip().split('\t')
            if '' in line_columns and not intent_over:
                intent_over = 1
            elif '' not in line_columns and not intent_over:
                assert len(
                    line_columns) == 2, "data_stats should have two columns"
                intent_stats[line_columns[0].strip()] = int(line_columns[1])
            elif '' not in line_columns and intent_over:
                assert len(
                    line_columns) == 2, "data_stats should have two columns"
                slot_stats[line_columns[0].strip()] = int(line_columns[1])

    return intent_stats, slot_stats


def get_slot_labels(slot_data,
                    utterance,
                    slot_stats,
                    tokenizer=naive_tokenizer):
    slot_labels = []

    original_words = tokenizer(utterance)
    slots_with_words = tokenizer(slot_data)
    i = j = 0
    while j < len(slots_with_words):
        if '<' in slots_with_words[j]:
            sep_num = 1
            while '</' not in slots_with_words[j + sep_num]:
                sep_num += 1

            left_slot_words = original_words[i:i + sep_num - 1]
            right_slot_words = slots_with_words[j + 1:j + sep_num]
            assert left_slot_words == right_slot_words, \
                "Words in the first column '%s' should match words '%s' in the third column" \
                % (original_words[i], slots_with_words[j + 1])

            slot_entity = slots_with_words[j][1:-1].strip()
            if '.' in slot_entity:
                slot_entity = slot_entity.split('.')[1].strip()
            j += (sep_num + 1)
            i += (sep_num - 1)

            if slot_entity in slot_stats:
                if sep_num > 2:
                    slot_labels.append('B-' + slot_entity)
                    slot_labels.extend([
                        'I-' + slot_entity,
                    ] * (sep_num - 2))
                else:
                    slot_labels.append('B-' + slot_entity)
            else:
                slot_labels.extend([
                    'O',
                ] * (sep_num - 1))
        else:
            assert original_words[i] == slots_with_words[j], \
                "Words in the first column '%s' should match words '%s' in the third column" \
                % (original_words[i], slots_with_words[j])
            slot_labels.append('O')
            j += 1
            i += 1
    return slot_labels


def data_classify(data_file,
                  intent_stats,
                  slot_stats,
                  tokenizer=naive_tokenizer):
    data = {}
    with open(data_file, 'r') as f:
        for line in f.readlines():
            line_columns = line.strip().split('\t')
            assert len(line_columns) == 3, \
                'Each line should only contain 3 type information(utterance, intent, slot).'
            utterance = line_columns[0].strip()
            utterance_words = tokenizer(utterance)
            intent = line_columns[1].strip()
            if len(intent.split()) > 1 or intent not in intent_stats:
                continue
            slot_labels = get_slot_labels(
                line_columns[2], utterance, slot_stats, tokenizer=tokenizer)
            assert len(utterance_words) == len(slot_labels), \
                'words must has a one2one mapping to entity label'
            if intent in data:
                data[intent].append([utterance_words, slot_labels])
            else:
                data[intent] = [
                    [utterance_words, slot_labels],
                ]
    return data


def union_classified_data(train_classified_data, test_classified_data):
    classified_data = {}
    for key in train_classified_data.keys():
        if key in test_classified_data:
            classified_data[
                key] = train_classified_data[key] + test_classified_data[key]
        else:
            classified_data[key] = train_classified_data[key]

    for key in test_classified_data.keys():
        if key not in classified_data:
            classified_data[key] = test_classified_data[key]

    return classified_data


def train_test_split(classified_data):
    train = {}
    test = {}
    for key in classified_data:
        num = len(classified_data[key])
        split_index = int(SPLIT_RATIO * num)
        train[key] = classified_data[key][:split_index]
        test[key] = classified_data[key][split_index:]

    return train, test


def build_dataset(train_data_file, test_data_file, data_stats_file):
    intent_stats, slot_stats = get_data_stats(data_stats_file)
    train_classified_data = data_classify(train_data_file, intent_stats,
                                          slot_stats)
    test_classified_data = data_classify(test_data_file, intent_stats,
                                         slot_stats)
    classified_data = union_classified_data(train_classified_data,
                                            test_classified_data)
    return train_test_split(classified_data)


def output_dataset(data, data_dir, train_or_test):
    output_intent = []
    output_utterance = []
    output_slots = []
    for key in data:
        output_intent.extend([
            key,
        ] * len(data[key]))
        for item in data[key]:
            output_utterance.append(item[0])
            output_slots.append(item[1])

    indexes = range(len(output_intent))
    random.shuffle(indexes)
    output_intent = [output_intent[i] for i in indexes]
    output_utterance = [output_utterance[i] for i in indexes]
    output_slots = [output_slots[i] for i in indexes]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, train_or_test + '.label'), 'w') as f:
        for intent in output_intent:
            f.write(intent + '\n')

    with open(os.path.join(data_dir, train_or_test + '.seq.in'), 'w') as f:
        for utterance in output_utterance:
            f.write(' '.join(utterance) + '\n')

    with open(os.path.join(data_dir, train_or_test + '.seq.out'), 'w') as f:
        for slots in output_slots:
            f.write(' '.join(slots) + '\n')


def main(argc, argv):
    if argc < 6:
        print('Usage:%s <train> <test> <data_stats> <train_dir> <test_dir>' %
              (argv[0]))
        exit(1)

    train, test = build_dataset(argv[1], argv[2], argv[3])
    output_dataset(train, argv[4], 'train')
    output_dataset(test, argv[5], 'test')


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
