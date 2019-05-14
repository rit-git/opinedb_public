import sys
import os

import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def read_results(dataset):
    fn = '%s_result.txt' % dataset

    methods = ['phrase model score',
               'marker model score',
               'IR-based',
               'AB method 0',
               'AB method 1',
               'AB method 2',
               'AB method 3',
               'Opine - marker, score',
               'Opine - marker, running time',
               'Opine - histogram, score',
               'Opine - histogram, running time']
    scores = { method : [] for method in methods}
    for line in open(fn):
        line = line.strip()
        score_id = None
        for method in methods:
            if line.startswith(method):
                score_id = method
                break

        if score_id != None:
            token = ''
            if '\t' in line:
                token = line.split('\t')[-1]
            elif ' ' in line:
                token = line.split(' ')[-1]
            scores[method].append(float(token))

    # table 1
    columns = []
    for difficulty in range(3):
        column = []
        N = len(scores['IR-based'])

        ir_based = [scores['IR-based'][i] for i in range(N) if i % 3 == difficulty]
        column.append(mean_confidence_interval(ir_based))
        for ab_mode in range(4):
            attr_name = 'AB method %d' % ab_mode
            score = [scores[attr_name][i] for i in range(N) if i % 3 == difficulty]
            column.append(mean_confidence_interval(score))
        opine = [scores['Opine - marker, score'][i] \
                for i in range(N) if i % 3 == difficulty]
        column.append(mean_confidence_interval(opine))
        columns.append(column)

    print('Table 1 dataset %s' % dataset)
    max_conf_int = 0.0
    for i in range(6):
        row = '\t'.join(['%.3f' % columns[j][i][0] for j in range(3)])
        max_conf_int = max(max_conf_int, max([columns[j][i][1] for j in range(3)]))
        print(row)
    print('max_conf_int =', max_conf_int)

    # table 2
    column = []
    LR_accuracy = mean_confidence_interval(scores['marker model score'])
    NDCG_10 = mean_confidence_interval(scores['Opine - marker, score'])
    runtime1 = mean_confidence_interval(scores['Opine - marker, running time'])
    column += [LR_accuracy, NDCG_10, runtime1]
    LR_accuracy = mean_confidence_interval(scores['phrase model score'])
    NDCG_10 = mean_confidence_interval(scores['Opine - histogram, score'])
    runtime2 = mean_confidence_interval(scores['Opine - histogram, running time'])
    column += [LR_accuracy, NDCG_10, runtime2]
    speedup = [b / a for (a, b) in zip(scores['Opine - marker, running time'],
                                       scores['Opine - histogram, running time'])]
    column.append(mean_confidence_interval(speedup))

    print('Table 2 dataset %s' % dataset)
    for i in range(7):
        print('%.2f %.3f' % (column[i][0], column[i][1]))
    print('avg. runtime - marker =', np.mean(scores['Opine - marker, running time']) / 100)
    print('avg. runtime - histogram =', np.mean(scores['Opine - histogram, running time']) / 100)


if __name__ == '__main__':
    datasets = ['london', 'amsterdam', 'toronto_lp', 'toronto_jp']

    for dataset in datasets:
        read_results(dataset)
