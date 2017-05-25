#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: build_fasttext_data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/24 21:22
"""
import os
import sys


def build_dataset(data_dir, train_or_test):
    data_fp = open(os.path.join(data_dir, train_or_test + '.seq.in'), 'r')
    label_fp = open(os.path.join(data_dir, train_or_test + '.label'), 'r')

    fasttext_data_fp = open(
        os.path.join(data_dir, train_or_test + '.fasttext'), 'w')
    while True:
        sample_x = data_fp.readline().strip()
        label = label_fp.readline().strip()
        if not sample_x:
            break
        fasttext_data_fp.write(sample_x + ' ' + '__label__' + label + '\n')

    data_fp.close()
    label_fp.close()
    fasttext_data_fp.close()


def main(argc, argv):
    if argc < 3:
        print('Usage:%s <train_dir> <test_dir>' % (argv[0]))
        exit(1)

    build_dataset(argv[1], 'train')
    build_dataset(argv[2], 'test')


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
