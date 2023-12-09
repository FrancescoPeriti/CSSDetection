import argparse
import torch
import random
import numpy as np
from stats import gold_scores, load_embeddings, apd, prt, ap, app, permutation_test
from tqdm import tqdm
from scipy import stats
from scipy.stats import spearmanr
from itertools import combinations
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
set_seed(42)

parser = argparse.ArgumentParser(
                    prog='Evaluate layer aggregation per LSC',
                    description='Compute Spearman Correlation for different aggregation')
parser.add_argument('--benchmark', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--layers', type=int)
parser.add_argument('--depth', type=int)
args = parser.parse_args()

# parameters
benchmark=args.benchmark
model=args.model_name
layers=args.layers
depth=args.depth

# load ground truth
y_true = gold_scores(benchmark)

# all possible layer aggregation
combs = list()
for i in range(2, depth + 1):
    combs += list(combinations(list(range(0, layers)), i))


# load embeddings from every layer
E = list()
for layer in range(1, layers + 1):
    E_layer = load_embeddings(benchmark, model, layer)
    for i, e in enumerate(E_layer):
        e = np.expand_dims(e[0], axis=0), np.expand_dims(e[1], axis=0)
        if layer == 1:
            E.append(e)
        else:
            lower_e = E[i]
            e0 = np.vstack([lower_e[0], e[0]])
            e1 = np.vstack([lower_e[1], e[1]])
            E[i] = (e0, e1)


# file header
header = 'benchmark\tmeasure\tmodel\tcomb\tspearman\tspearmanp\tpermutation_test\tpermutation_testp'

# file output
stats_file = "comb_stats.tsv"
if not Path(stats_file).is_file():
    lines = [header + '\n']
else:
    lines = open(stats_file, mode='r', encoding='utf-8').readlines()

# -- Results wrapper --
results = list()
for comb in tqdm(combs, desc='Combining embeddings'):
    for agg in ['sum', 'concat']:
        comb_name = f'{agg}_{"-".join([str(l) for l in comb])}'
        if agg == 'sum':
            agg_E = [(e[0][comb, :, :].sum(axis=0), e[1][comb, :, :].sum(axis=0)) for e in E]
        elif agg == 'concat':
            stack = lambda x: x.transpose(1, 0, 2).reshape(x.shape[1], -1)
            agg_E = [(stack(e[0][comb, :, :]), stack(e[1][comb, :, :])) for e in E]

        y = apd(agg_E)
        sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
        lines.append(f'{benchmark}\tapd\t{model}\t{comb_name}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')

        y = prt(agg_E)
        sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
        lines.append(f'{benchmark}\tprt\t{model}\t{comb_name}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')

        # AP + jsd
        y = ap(agg_E, metric='jsd')
        sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
        lines.append(
            f'{benchmark}\tap+jsd\t{model}\t{comb_name}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')

        # AP + APDP
        y = ap(agg_E, metric='apdp')
        sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
        lines.append(
            f'{benchmark}\tap+apdp\t{model}\t{comb_name}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')

        # AP + JSD
        y = app(agg_E, metric='jsd')
        sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
        lines.append(
            f'{benchmark}\tapp+jsd\t{model}\t{comb_name}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')

        # AP + APDP
        y = app(agg_E, metric='apdp')
        sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
        lines.append(
            f'{benchmark}\tapp+apdp\t{model}\t{comb_name}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')

with open(stats_file, mode='w', encoding='utf-8') as f:
    f.writelines(lines)
