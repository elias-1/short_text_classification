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
import sys

import matplotlib.pyplot as plt
import numpy as np


def main(argc, argv):
    grid = np.array([
        [[
            2.48168e-06, 7.20005e-07, 1.18812e-06, 1.188e-06, 0.000129729,
            0.000239056, 0.0327365, 0.102477, 0.864373, 3.56355e-06,
            3.56356e-06, 3.56356e-06, 3.56355e-06, 3.56355e-06, 3.56355e-06,
            3.56356e-06, 3.56356e-06, 3.56356e-06, 3.56355e-06, 3.56356e-06
        ]],
        [[
            2.81721e-06, 5.21519e-06, 6.07923e-06, 2.40766e-05, 0.00024242,
            0.000531199, 0.0647686, 0.000360519, 0.195387, 0.000369511,
            0.738191, 1.23509e-05, 1.23509e-05, 1.23509e-05, 1.23509e-05,
            1.23509e-05, 1.23509e-05, 1.23509e-05, 1.23509e-05, 1.23509e-05
        ]],
        [[
            1.04009e-06, 2.28867e-06, 1.2288e-06, 1.4493e-06, 0.000208147,
            0.00037682, 0.0382299, 0.106703, 0.854437, 3.54277e-06,
            3.54277e-06, 3.54277e-06, 3.54277e-06, 3.54277e-06, 3.54277e-06,
            3.54277e-06, 3.54277e-06, 3.54277e-06, 3.54277e-06, 3.54277e-06
        ]],
        [[
            4.76718e-05, 8.15489e-06, 0.00200974, 0.00624807, 2.07804e-05,
            0.991665, 3.02897e-08, 3.28187e-07, 7.24383e-09, 3.65829e-08,
            3.65829e-08, 3.65829e-08, 3.65829e-08, 3.65829e-08, 3.65829e-08,
            3.65829e-08, 3.65829e-08, 3.65829e-08, 3.65829e-08, 3.65829e-08
        ]],
    ])

    seq_len = 9

    plt.figure()
    ax1 = plt.gca()

    ax1.imshow(grid[3][:seq_len], extent=[0, seq_len, 0, 1])
    ax1.set_title('Default')

    # ax2.imshow(grid, extent=[0, 100, 0, 1], aspect='auto')
    # ax2.set_title('Auto-scaled Aspect')
    #
    # ax3.imshow(grid, extent=[0, 100, 0, 1], aspect=100)
    # ax3.set_title('Manually Set Aspect')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
