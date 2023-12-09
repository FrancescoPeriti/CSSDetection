import os
import argparse
from app import *
from scipy.spatial.distance import cdist, cosine
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from scipy import stats
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.cluster import AffinityPropagation
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon 
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import unicodedata


def gold_scores(benchmark:str) -> np.array:
    f = 'tsv' if 'dwug_no' in benchmark else 'csv'
    
    if 'dwug_no' in benchmark or 'dwug_ru' in benchmark:
        df = pd.read_csv(f'{benchmark}/stats/stats_groupings.{f}', sep='\t', encoding='utf-8')
    else:
        df = pd.read_csv(f'{benchmark}/stats/opt/stats_groupings.{f}', sep='\t', encoding='utf-8')

    df['lemma'] = df['lemma'].str.normalize('NFKD') # different codings between os.listdir and pandas for spanish targets
    df = df.sort_values(by='lemma')
    
    if 'dwug_ru' in benchmark:
        return np.array([-row['COMPARE'] for _, row in df.iterrows()])
    else:
        return np.array([row['change_graded'] for _, row in df.iterrows()])

def load_embeddings(benchmark:str, model:str, layer:str) -> list:
    E = list()
    targets = sorted(os.listdir(f'{benchmark}/data'))
    
    for target in targets:
        E1 = np.load(f'{benchmark}/embeddings/{model}/corpus1/{layer}/{target}.npy')
        E2 = np.load(f'{benchmark}/embeddings/{model}/corpus2/{layer}/{target}.npy') 
        E.append((E1, E2))
    
    return E

def dist(a:np.array, b:np.array, metric:str='cosine') -> np.array:
    d = cdist(a, b, metric)
    if np.isnan(d).any(): # sometimes cdist results in nan values.. But it's faster
        return pairwise_distances(a, b, metric)
    return d

def apd(E:list, metric:str='cosine') -> np.array:
    return np.array([np.mean(dist(e[0], e[1], metric=metric)) for e in E])

def prt(E:list) -> np.array:
    return np.array([cosine(e[0].mean(axis=0), e[1].mean(axis=0)) for e in E])

def ap(E:list, metric:str='jsd') -> np.array:
    ap = AffinityPropagation(affinity='precomputed', 
                         damping=0.9,
                         max_iter=200,
                         convergence_iter=15,
                         copy=True,
                         preference=None,
                         random_state=42)
    
    y = list()
    for e in E:
        e12 = np.concatenate([e[0], e[1]], axis=0)
        sim = cosine_similarity(e12)
        ap.fit(sim)

        if metric == 'jsd':
            y.append(jsd(e, ap))
        if metric == 'apdp':
            y.append(apdp(e, ap))
    return np.array(y)

def app(E:list, metric:str='jsd') -> np.array:
    
    y = list()
    for e in E:
        app = APosterioriaffinityPropagation(affinity='cosine', 
                               damping=0.9,
                               max_iter=200,
                               convergence_iter=15,
                               copy=True,
                               preference=None,
                               random_state=42)
        app.fit(e[0])
        app.fit(e[1])

        if metric == 'jsd':
            y.append(jsd(e, app))
        if metric == 'apdp':
            y.append(apdp(e, app))
    return np.array(y)

def jsd(embeddings:list, ap) -> float:
    L = ap.labels_
    L1, L2 = L[:embeddings[0].shape[0]], L[embeddings[0].shape[0]:]   
    labels = np.unique(np.concatenate([L1, L2]))

    c1 = Counter(L1)
    c2 = Counter(L2)

    L1_dist = np.array([c1[l] for l in labels])
    L2_dist = np.array([c2[l] for l in labels])
    
    L1_dist = L1_dist / L1_dist.sum()
    L2_dist = L2_dist / L2_dist.sum()
    
    return jensenshannon(L1_dist, L2_dist)    


def apdp(embeddings:list, ap, metric:str='canberra') -> float:
    L = ap.labels_
    L1, L2 = L[:embeddings[0].shape[0]], L[embeddings[0].shape[0]:]
    # cluster centroids
    mu_E1 = np.array([embeddings[0][L1 == label].mean(axis=0) for label in np.unique(L1)])
    mu_E2 = np.array([embeddings[1][L2 == label].mean(axis=0) for label in np.unique(L2)])
    return np.mean(cdist(mu_E1, mu_E2, metric=metric))

def permutation_test(y_true: np.array, y: np.array, n_resamples: int = 100) -> float:
    def spearmanr_statistic(y):  # permute only `x`
        return spearmanr(y, y_true).correlation

    res = stats.permutation_test((y,), spearmanr_statistic, n_resamples=n_resamples,
                                       permutation_type='pairings')
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Evaluate LSC',
        description='Compute Spearman Correlation for APD-PRT-APP-AP')
    parser.add_argument('--benchmark', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--layer', type=str)
    args = parser.parse_args()


    # parameters
    benchmark=args.benchmark
    model=args.model_name
    layer=args.layer

    
    # file header
    header = 'benchmark\tmeasure\tmodel\tlayer\tspearman\tspearmanp\tpermutation_test\tpermutation_testp'

    # file output
    stats_file = "stats.tsv"
    if not Path(stats_file).is_file():
        lines = [header+ '\n']
    else:
        lines = open(stats_file, mode='r', encoding='utf-8').readlines()

    # load ground truth
    y_true = gold_scores(benchmark)

    # load embeddings
    E = load_embeddings(benchmark, model, layer)
    
    # APD
    y = apd(E)
    sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
    lines.append(f'{benchmark}\tapd\t{model}\t{layer}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')
    
    # PRT
    y = prt(E)
    sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
    lines.append(f'{benchmark}\tprt\t{model}\t{layer}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')
    
    # AP + jsd
    y = ap(E, metric='jsd')
    sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
    lines.append(f'{benchmark}\tap+jsd\t{model}\t{layer}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')
    
    # AP + APDP
    y = ap(E, metric='apdp')
    sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
    lines.append(f'{benchmark}\tap+apdp\t{model}\t{layer}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')
    
    # AP + JSD
    y = app(E, metric='jsd')
    sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
    lines.append(f'{benchmark}\tapp+jsd\t{model}\t{layer}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')
    
    # AP + APDP
    y = app(E, metric='apdp')
    sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
    lines.append(f'{benchmark}\tapp+apdp\t{model}\t{layer}\t{sp.correlation.round(3)}\t{sp.pvalue.round(3)}\t{pt.statistic.round(3)}\t{pt.pvalue.round(3)}\n')
    
    with open(stats_file, mode='w', encoding='utf-8') as f:
        f.writelines(lines)
