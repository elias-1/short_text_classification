#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: vocab_size_stat.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/23 20:09
"""
import sys


def main(argc, argv):
    if argc < 3:
        print('Usage:%s <train_file> <max_sentence_len>' % (argv[0]))
        exit(1)

    max_sentence_len = int(argv[2])
    vocab = {}
    with open(argv[1], 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()[:max_sentence_len]
            for token in tokens:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1

    print('vocabulary size: %d' % len(vocab))


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
