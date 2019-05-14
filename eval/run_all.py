import random
import sys
import os

if __name__ == '__main__':
    random.seed(123)
    rep = 10
    datasets = ['amsterdam', 'london', 'toronto_jp', 'toronto_lp']

    for dataset in datasets:
        os.system('rm %s_result.txt' % dataset)
        for _ in range(rep):
            seed = random.randint(0, 1000000000)
            os.system('python evaluate.py %s %d >> %s_result.txt' % (dataset, seed, dataset))
