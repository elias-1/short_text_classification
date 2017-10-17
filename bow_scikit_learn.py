#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: bow_scikit_learn.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/27 22:56
"""
import os
import sys

import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def stats_class(data_dir):
    classes = {}
    with open(os.path.join(data_dir, 'train.label'), 'r') as f:
        for line in f.readlines():
            if line.strip() not in classes:
                classes[line.strip()] = len(classes)
    return classes


def build_dataset(data_dir, class_stats, train_or_test):
    data_fp = open(os.path.join(data_dir, train_or_test + '.seq.in'), 'r')
    label_fp = open(os.path.join(data_dir, train_or_test + '.label'), 'r')

    data = []
    y = []
    while True:
        sample_x = data_fp.readline().strip()
        label = label_fp.readline().strip()
        if not sample_x:
            break
        data.append(sample_x)
        y.append(class_stats[label])

    data_fp.close()
    label_fp.close()
    return data, y


def my_tokenizer(s):
    return s.split(' ')


def process_pipeline():
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(train_x)
    # tfidf_transformer = TfidfTransformer(use_idf=False)
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # clf = MultinomialNB().fit(X_train_tfidf, train_y)

    text_clf = Pipeline([
        ('vect', CountVectorizer(tokenizer=my_tokenizer)),
        ('tfidf', TfidfTransformer()),
        # ('clf', MultinomialNB()),
        ('clf', SGDClassifier(
            loss='hinge', penalty='l2', n_iter=5, random_state=42)),
    ])
    return text_clf


def do_test(text_clf, test_x, test_y, test_label_names):
    predicted = text_clf.predict(test_x)
    print(np.mean(predicted == test_y))

    print(metrics.classification_report(
        test_y, predicted, target_names=test_label_names))
    metrics.confusion_matrix(test_y, predicted)


def grid_search(text_clf, train_x, train_y):
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_x, train_y)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # print(gs_clf.cv_results_)
    return gs_clf


def main(argc, argv):
    if argc < 3:
        print('Usage:%s <train_dir> <test_dir>' % (argv[0]))
        exit(1)

    class_stats = stats_class(argv[1])
    train_x, train_y = build_dataset(argv[1], class_stats, 'train')
    test_x, test_y = build_dataset(argv[2], class_stats, 'test')

    text_clf = process_pipeline()
    gs_clf = grid_search(text_clf, train_x, train_y)
    int2class = {}
    for key in class_stats.keys():
        int2class[class_stats[key]] = key
    test_name = [int2class[y] for y in test_y]
    do_test(gs_clf, test_x, test_y, test_name)


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
