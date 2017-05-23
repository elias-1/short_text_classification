#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: draw.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/23 20:39
"""

import matplotlib.pyplot as plt
import numpy as np

retrain_num = 200


def smooth(data):
    ratio = 0.1
    data_new = data[data > 0]
    for i in range(1, len(data) - 1):
        data_new[i] += ratio * np.sum(data[i - 1:i + 2])
        data_new[i] /= 1 + 3 * ratio
        # while data_new[i] > 0.8:
        #     data_new[i] -= 0.001

    return data_new


date = '20170519-ehr-'
batch_size = 32

data_path_list = [
    ['../data/acc/{:s}random.npy'.format(date), 'g.:'],
    ['../data/acc/{:s}entropy-max.npy'.format(date), 'b|:'],
    ['../data/acc/{:s}batch-mode.npy'.format(date), 'm*-'],
    # ['../data/acc/{:s}exploration-exploitation.npy'.format(date),'k^-'],

    # ['../data/acc/{:s}batch-mode-pairwise.npy'.format(date),'rs-'],
    [
        '../data/acc/{:s}exploration-exploitation-pairwise.npy'.format(date),
        'r|-'
    ],
]

ax = plt.subplot(111)
for data_path, color in data_path_list:
    # data_path = data_path.replace('.npy','-retrain-{:d}.npy'.format(retrain_num))
    try:
        data = np.load(data_path)
        print data.shape,
    except:
        print '时间可能不对'
        continue
    for i in range(len(data)):
        if np.sum(data[i]) < 1:
            data = data[:i]
            print i
            break
    y = np.mean(data, axis=0)
    # print y.shape
    y = smooth(y)
    x = [(i + 1) * batch_size for i in range(len(y))]
    label = data_path.split('/')[-1].split('_')[-1].replace(
        '.npy', '').split('mnist-')[-1]
    label = label.split('ehr-')[1]
    print label, len(y), max(y)
    line = ax.plot(x, y, color, label=label)
    plt.grid(True)
    plt.xlabel('Training Set Size')
    plt.ylabel('Classification Accuracy')
    plt.ylim(0.55, 0.85)
    # plt.xlim(288)
    ax.legend(loc='lower right')
    plt.title('Heart Failure Prediction')
save_path = '../result/hf-acc.eps'
save_path = '../result/hf-acc-bmal.eps'
save_path = '../result/hf-acc-ee.eps'
save_path = '../result/hf-acc-base.eps'
# save_path = '/home/cc/win7/p2.png'
plt.savefig(save_path)
plt.show()
