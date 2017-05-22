#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: func_utils.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017-5-22 09:19:59
"""

import numpy as np

MAX_SENTENCE_LEN = 20
MAX_COMMON_LEN = 5
ENTITY_TYPES = ['@d@', '@s@', '@l@', '@o@', '@m@', '@dp@', '@bp@']
"""ENTITY_TYPES
len([PAD, O]) + len(ENTITY_TYPES) * len([B I])
"""


def load_data(path, max_sentence_len):
    wx = []
    y = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        assert (len(ss) == (max_sentence_len + 1)), \
            "[line:%d]len ss:%d,origin len:%d\n%s" % (ln, len(ss), len(line), line)

        lwx = []
        for i in range(max_sentence_len):
            lwx.append(int(ss[i]))

        wx.append(lwx)
        y.append(int(ss[max_sentence_len]))
    fp.close()
    return np.array(wx), np.array(y)


def load_data_dkgam(path, max_sentence_len, max_replace_entity_nums):
    wx = []
    y = []
    entity_info = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        assert (len(ss) == (max_sentence_len + 1 + max_replace_entity_nums)), \
            "[line:%d]len ss:%d,origin len:%d\n%s" % (ln, len(ss), len(line), line)
        lwx = []
        lentity_info = []
        for i in range(max_sentence_len):
            lwx.append(int(ss[i]))

        len_features = max_sentence_len
        for i in range(max_replace_entity_nums):
            lentity_info.append(int(ss[len_features + 1 + i]))

        wx.append(lwx)
        entity_info.append(lentity_info)
        y.append(int(ss[max_sentence_len]))
    fp.close()
    return np.array(wx), np.array(y), np.array(entity_info)


def load_data_mt_dkgam(path, max_sentence_len, max_replace_entity_nums):
    wx = []
    ner_y = []
    clfier_y = []
    entity_info = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        if len(ss) != (max_sentence_len * 2 + 1 + max_replace_entity_nums):
            print("[line:%d]len ss:%d,origin len:%d\n%s" % (ln, len(ss),
                                                            len(line), line))
        assert (len(ss) == (max_sentence_len * 2 + 1 + max_replace_entity_nums)), \
            "[line:%d]len ss:%d,origin len:%d\n%s" % (ln, len(ss), len(line), line)
        lwx = []
        lentity_info = []
        lner_y = []
        for i in range(max_sentence_len):
            lwx.append(int(ss[i]))

        for i in range(max_sentence_len):
            lner_y.append(int(ss[max_sentence_len + i]))

        for i in range(max_replace_entity_nums):
            lentity_info.append(int(ss[max_sentence_len * 2 + 1 + i]))

        wx.append(lwx)
        ner_y.append(lner_y)
        entity_info.append(lentity_info)
        clfier_y.append(int(ss[max_sentence_len * 2]))
    fp.close()
    return np.array(wx), np.array(ner_y), np.array(wx), np.array(
        clfier_y), np.array(entity_info)
