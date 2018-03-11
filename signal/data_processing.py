#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 All Rights Reserved
#
"""
File: data_processing.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 11/03/2018 9:15 AM
"""


import csv
import argparse
import random
from itertools import chain

from func_utils import MIN_SEQ_LEN, MAX_SEQ_LEN, SIGNAL_WIDTH


def arg_parser():
    parser = argparse.ArgumentParser(
        prog='prepare_data',
        formatter_class=argparse.RawTextHelpFormatter,
        description='Prepare data for input of deep model')

    parser.add_argument(
        '-x_data',
        dest='x_data',
        default='Sample.csv',
        type=str,
        help='x data')

    parser.add_argument(
        '-y_data',
        dest='y_data',
        default='Label.csv',
        type=str,
        help='y data')

    parser.add_argument(
        '-train_data',
        dest='train_data',
        default='train_data.csv',
        type=str,
        help='train_data')

    parser.add_argument(
        '-test_data',
        dest='test_data',
        default='test_data.csv',
        type=str,
        help='test_data')

    parser.add_argument(
        '-train_ratio',
        dest='train_ratio',
        default=0.8,
        type=float,
        help='test_data')

    return parser

def main(args):
    x_data = []
    with open(args.x_data) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            x_data.append(row)

    assert len(x_data[0]) == SIGNAL_WIDTH, 'Please make sure the signal width in func_utils.py'

    y_data = []
    with open(args.y_data) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            y_data.extend(row)

    min_seq_len = MIN_SEQ_LEN
    max_seq_len = MAX_SEQ_LEN

    formarted_data = []
    i = 0
    while True:
        current_seq_len = random.randint(min_seq_len, max_seq_len)
        if i + current_seq_len > len(x_data):
            break

        current_formart = list(chain.from_iterable(x_data[i:i+current_seq_len]))
        current_formart += ['0'] * len(x_data[0]) * (max_seq_len-current_seq_len)

        current_formart += y_data[i:i+current_seq_len]
        current_formart += ['0'] * (max_seq_len-current_seq_len)
        formarted_data.append(current_formart)
        i += current_seq_len

    train_len = int(args.train_ratio * len(formarted_data))
    with open(args.train_data, 'w+') as f:
        csv_writer = csv.writer(f, delimiter=' ')
        csv_writer.writerows(formarted_data[:train_len])

    with open(args.test_data, 'w+') as f:
        csv_writer = csv.writer(f, delimiter=' ')
        csv_writer.writerows(formarted_data[train_len:])




if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    main(args)