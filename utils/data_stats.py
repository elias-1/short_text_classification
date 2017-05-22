#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: prepare_data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/15 21:01

Usage: 
python data_stats.py ../data/atis.train.tsv ../data/atis.test.tsv entity data_stats.txt 25 15
"""

import re
import sys
from copy import deepcopy

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def naive_tokenizer(sentence):
    """Naive tokenizer: split the sentence by space into a list of tokens."""
    return sentence.split()


def data_stats(file_name, slot_or_entity='entity', tokenizer=naive_tokenizer):
    multi_label = 0
    slot_stats = {}
    intent_stats = {}

    with open(file_name, 'r') as f:
        line_num = 1
        for line in f.readlines():
            print('processing line: %d' % line_num)
            line_columns = line.strip().split('\t')
            if len(line_columns[1].strip().split()) > 1:
                multi_label += 1
                continue

            if line_columns[1].strip() in intent_stats:
                intent_stats[line_columns[1].strip()] += 1
            else:
                intent_stats[line_columns[1].strip()] = 1

            assert len(line_columns) == 3, \
                'Each line should only contain 3 type information(utterance, intent, slot).'

            original_words = tokenizer(line_columns[0])
            slots_with_words = tokenizer(line_columns[2])
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

                    if slot_or_entity not in ['slot', 'entity']:
                        raise ValueError(
                            "slot_or_entity must be 'slot' or 'entity'")

                    slot_entity = slots_with_words[j][1:-1]
                    if slot_or_entity == 'entity':
                        if '.' in slot_entity:
                            slot_entity = slot_entity.split('.')[1]

                    if slot_entity in slot_stats:
                        slot_stats[slot_entity] += 1
                    else:
                        slot_stats[slot_entity] = 1

                    j += (sep_num + 1)
                    i += (sep_num - 1)
                else:
                    assert original_words[i] == slots_with_words[j], \
                        "Words in the first column '%s' should match words '%s' in the third column" \
                         % (original_words[i], slots_with_words[j])
                    j += 1
                    i += 1
            line_num += 1
    return slot_stats, multi_label, intent_stats


def merge_dict(train_slot_stats, test_slot_stats):
    total_slot_stats = deepcopy(train_slot_stats)
    for key in test_slot_stats.keys():
        if key in total_slot_stats:
            total_slot_stats[key] += test_slot_stats[key]
        else:
            total_slot_stats[key] = test_slot_stats[key]
    return total_slot_stats


def sorted_for_print(stats, str_for_print):
    stats_list = list(map(lambda x, y: (x, y), stats.keys(), stats.values()))
    stats_list_sorted = sorted(
        stats_list, key=lambda item: item[1], reverse=True)
    stats_str_sorted = list(
        map(lambda x: str(x[1]) + '\t' + x[0], stats_list_sorted))
    print(str_for_print)
    print '\n'.join(stats_str_sorted)
    return stats_list_sorted


def main(argc, argv):
    if argc < 7:
        print(
            'Usage:%s <train> <test> <slot/entity> <stats_output> <entity threshold> <intent threshold>'
            % (argv[0]))
        exit(1)

    train_slot_stats, train_multi_label, train_intent_stats = data_stats(
        argv[1], argv[3])
    test_slot_stats, test_multi_label, test_intent_stats = data_stats(
        argv[2], argv[3])
    total_slot_stats = merge_dict(train_slot_stats, test_slot_stats)
    total_intent_stats = merge_dict(train_intent_stats, test_intent_stats)

    sorted_for_print(train_slot_stats, 'slot stats for train data')
    sorted_for_print(test_slot_stats, 'slot stats for test data')
    slot_stats_list = sorted_for_print(total_slot_stats,
                                       'slot stats for total data')

    sorted_for_print(train_intent_stats, 'intent stats for train data')
    sorted_for_print(test_intent_stats, 'intent stats for test data')
    intent_stats_list = sorted_for_print(total_intent_stats,
                                         'intent stats for total data')

    print('there exists %d multi label utterance in training set' %
          train_multi_label)
    print(
        'there exists %d multi label utterance in test set' % test_multi_label)
    print('there exists %d multi label utterance in whole set' %
          (train_multi_label + test_multi_label))

    entity_threshold = int(argv[5])
    intent_threshold = int(argv[6])
    with open(argv[4], 'w') as f:
        for item in intent_stats_list:
            if item[1] > intent_threshold:
                f.write(item[0] + '\t' + str(item[1]))
                f.write('\n')
        f.write('\n')
        for item in slot_stats_list:
            if item[1] > entity_threshold:
                f.write(item[0] + '\t' + str(item[1]))
                f.write('\n')


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
