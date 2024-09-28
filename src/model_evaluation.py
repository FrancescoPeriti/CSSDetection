import warnings

#suppress warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
import os
import csv
import dill
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score, homogeneity_score, fowlkes_mallows_score, adjusted_mutual_info_score
from sklearn import metrics
from correlation_clustering import cluster_correlation_search
from collections import defaultdict
from scipy.spatial.distance import jensenshannon

def auto_discretize(arr, num_bins=4):
    # Determine bin edges based on the distribution of values in the array
    bin_edges = np.histogram_bin_edges(arr, bins=num_bins)

    # Discretize the values in 'arr' into the determined bins and assign integers
    result = np.digitize(arr, bin_edges, right=True)

    return result

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def load_gold_lsc(benchmark):
    if benchmark.startswith('dwug_ru'):
        df = pd.read_csv(f'{benchmark}/stats/stats_groupings.csv', sep='\t')
        df = df.rename(columns={'COMPARE':'change_graded'})
    elif benchmark.startswith('dwug_no'):
        df = pd.read_csv(f'{benchmark}/stats/stats_groupings.tsv', sep='\t')
    else:
        df = pd.read_csv(f'{benchmark}/stats/opt/stats_groupings.csv', sep='\t')
    
    return df

def load_word_usages(benchmark, target):
    df = pd.read_csv(f'{benchmark}/data/{target}/uses.csv', sep='\t', quoting=csv.QUOTE_NONE)
    
    if 'dwug_ru12' in benchmark:
        df['grouping'] = [1 if '1700-1916' == i else 2 for i in df['grouping']]
    elif 'dwug_ru23' in benchmark:
        df['grouping'] = [1 if '1918-1990' == i else 2 for i in df['grouping']]
    elif 'dwug_ru13' in benchmark:
        df['grouping'] = [1 if '1700-1916' == i else 2 for i in df['grouping']]
    elif 'dwug_no12' in benchmark:
        df['grouping'] = [1 if '1970-2015' == i else 2 for i in df['grouping']]
    elif 'dwug_no23' in benchmark:
        df['grouping'] = [1 if '1980-1990' == i else 2 for i in df['grouping']]
    
    if 'dwug_ru' in benchmark:
        df['identifier'] = [str(i).split('_')[-1] for i in df['identifier']]
    
    return df

def read_data(benchmark, model, layer, wsi=True):
    # read targets
    targets = sorted(os.listdir(f'{benchmark}/data/'))
    
    # read judgments by LM
    dfs = dill.load(open(f'scores/{benchmark}/{model}/{layer}.dill', mode='rb'))
    targets = [target for target in targets if target in dfs]
    dfs = [pd.DataFrame(dfs[target]) for target in targets]
    
    # pre-processing
    for df in dfs:
        df['identifier1'] = df['identifier1'].apply(lambda x: x.replace('Name: identifier1', ''))
        df['identifier2'] = df['identifier2'].apply(lambda x: x.replace('Name: identifier2', ''))   
    
    gold_wic_dfs = list()
    gold_wsi_dfs = list()
    for i, target in enumerate(targets):

        # read gold judgments
        if 'dwug_no' in benchmark:
            gold_wic_df = pd.read_csv(f'{benchmark}/data_joint/data_joint.tsv', sep='\t')
            gold_wic_df = gold_wic_df[gold_wic_df['lemma']==target][['identifier1', 'identifier2', 'judgment']]
        else:
            gold_wic_df = pd.read_csv(f'{benchmark}/data/{target}/judgments.csv', sep='\t')[['identifier1', 'identifier2', 'judgment']]

        gold_wic_df['identifier1'] = gold_wic_df['identifier1'].astype(str)
        gold_wic_df['identifier2'] = gold_wic_df['identifier2'].astype(str)
        gold_wic_df = gold_wic_df.sort_values(['identifier1', 'identifier2']).groupby(['identifier1', 'identifier2']).mean().reset_index()
        
        # ignore noisy word usages (i.e. cluster = -1)
        if wsi:
            if benchmark.startswith('dwug_no'):
                gold_wsi_df = pd.read_csv(f'{benchmark}/clusters/{target}.tsv', sep='\t')
            else:
                gold_wsi_df = pd.read_csv(f'{benchmark}/clusters/opt/{target}.csv', sep='\t')
            valid_idx = gold_wsi_df[gold_wsi_df.cluster!=-1].identifier.values
            gold_wsi_df = gold_wsi_df[gold_wsi_df.identifier.isin(valid_idx)]
            gold_wic_df = gold_wic_df[gold_wic_df.identifier1.isin(valid_idx) & gold_wic_df.identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2']).reset_index()
            dfs[i] = dfs[i][dfs[i].identifier1.isin(valid_idx) & dfs[i].identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2'])
        
        # ignore usage pairs for which it wasn't possible to make a prediction
        usages = set(dfs[i].identifier1.tolist()+dfs[i].identifier2.tolist())
        gold_wic_df = gold_wic_df[gold_wic_df.identifier1.isin(usages) & gold_wic_df.identifier2.isin(usages)]
        if wsi:
            gold_wsi_df = gold_wsi_df[gold_wsi_df.identifier.isin(usages)]
        
        # add grouping info
        uses = load_word_usages(benchmark, target)[['identifier', 'grouping']]
        uses = {row['identifier']: row['grouping'] for _, row in uses.iterrows()}
        gold_wic_df['grouping1'] = [uses[row['identifier1']] for _, row in gold_wic_df.iterrows()]
        gold_wic_df['grouping2'] = [uses[row['identifier2']] for _, row in gold_wic_df.iterrows()]
        dfs[i]['grouping1'] = [uses[row['identifier1']] for _, row in dfs[i].iterrows()]
        dfs[i]['grouping2'] = [uses[row['identifier2']] for _, row in dfs[i].iterrows()]
        
        gold_wic_dfs.append(gold_wic_df)
        
        if wsi:
            gold_wsi_dfs.append(gold_wsi_df)
    
    gold_lsc_df = load_gold_lsc(benchmark)[['lemma', 'change_graded']]
    if 'dwug_no' not in benchmark:
        gold_lsc_df['lemma'] = gold_lsc_df['lemma'].str.normalize('NFKD') 
    gold_lsc_df = gold_lsc_df[gold_lsc_df['lemma'].isin(targets)].sort_values('lemma')
    print(gold_lsc_df.shape, len(targets))
    
    return dfs, gold_wic_dfs, gold_wsi_dfs if wsi else None, gold_lsc_df

def read_data_chatgpt(benchmark, wsi=True):
    dfs = list()
    gold_wic_dfs = list()
    gold_wsi_dfs = list()
    
    # read targets
    targets = sorted(os.listdir(f'{benchmark}/data/'))
    for target in targets:
        uses = load_word_usages(benchmark, target)

        # judgments
        gold_wic_df = pd.read_csv(f'{benchmark}/data/{target}/judgments.csv', sep='\t')
        gold_wic_df['identifier1'] = gold_wic_df['identifier1'].astype(str)
        gold_wic_df['identifier2'] = gold_wic_df['identifier2'].astype(str)

        # remove useless columns
        uses = uses[['identifier', 'context', 'indexes_target_token', 'grouping']]
        gold_wic_df = gold_wic_df[['identifier1', 'identifier2', 'annotator', 'judgment']]
        
        # join the two dataframes
        gold_wic_df = gold_wic_df.merge(uses, left_on='identifier1', right_on='identifier')
        gold_wic_df = gold_wic_df.rename(
            columns={'context': 'context1', 'indexes_target_token': 'indexes_target_token1', 'grouping': 'grouping1'})
        del gold_wic_df['identifier']
        gold_wic_df = gold_wic_df.merge(uses, left_on='identifier2', right_on='identifier')
        gold_wic_df = gold_wic_df.rename(
            columns={'context': 'context2', 'indexes_target_token': 'indexes_target_token2', 'grouping': 'grouping2'})
        del gold_wic_df['identifier']

        # take the mean of annotators
        gold_wic_df = gold_wic_df.groupby(
            ['identifier1', 'identifier2', 'context1', 'context2', 'indexes_target_token1', 'indexes_target_token2',
             'grouping1', 'grouping2']).mean().reset_index()
        
        judgments = np.load(f'chatgpt_annotation/{benchmark}/{target}.npy')
        df = gold_wic_df.copy()
        df['judgment'] = judgments

        # ignore noisy word usages (i.e. cluster = -1)
        if wsi:
            gold_wsi_df = pd.read_csv(f'{benchmark}/clusters/opt/{target}.csv', sep='\t')
            valid_idx = gold_wsi_df[gold_wsi_df.cluster!=-1].identifier.values
            gold_wsi_df = gold_wsi_df[gold_wsi_df.identifier.isin(valid_idx)]
            gold_wic_df = gold_wic_df[gold_wic_df.identifier1.isin(valid_idx) & gold_wic_df.identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2']).reset_index()
            df = df[df.identifier1.isin(valid_idx) & df.identifier2.isin(valid_idx)].sort_values(['identifier1', 'identifier2'])

        dfs.append(df)
        gold_wic_dfs.append(gold_wic_df)
        
        if wsi:
            gold_wsi_dfs.append(gold_wsi_df)
    
    gold_lsc_df = load_gold_lsc(benchmark)[['lemma', 'change_graded']]
    
    return dfs, gold_wic_dfs, gold_wsi_dfs if wsi else None, gold_lsc_df
        
def WiC(dfs, gold_dfs):
    # store gold judgments and predicted judgments
    y, y_true = list(), list()
    for df, gold_df in zip(dfs, gold_dfs):
        y.extend(df.judgment.values.tolist())
        y_true.extend(gold_df.judgment.values.tolist())    
    
    return spearmanr(y, y_true)

def WSI(dfs, gold_wsi_dfs):
    # wrappers
    clusters_dists = list() # cluster distributions
    clusters_labels = list() # cluster labels
    clusters_metrics = defaultdict(list) # cluster metrics

    # concat datasets - standardize judgments - and then split again
    start_end = [(0, dfs[0].shape[0])] + [(pd.concat(dfs[:i]).shape[0], pd.concat(dfs[:i]).shape[0]+dfs[i].shape[0]) for i in range(1, len(dfs))]
    dfs = pd.concat(dfs).reset_index(drop=True)
    #dfs['judgment'] = auto_discretize(dfs['judgment'].values)
    dfs['judgment'] = (dfs['judgment'].values-dfs['judgment'].values.mean())/dfs['judgment'].values.std()
    dfs = [dfs.iloc[idx[0]:idx[1]] for idx in start_end]
    
    for i, df in enumerate(dfs):
        # DWUG
        graph = nx.Graph()
        for _, row in df.iterrows():
            graph.add_edge(row['identifier1'] + '###' + str(row['grouping1']),
                           row['identifier2'] + '###' + str(row['grouping2']),
                           weight=row['judgment'])

        classes = []
        for init in range(1):
            classes, _ = cluster_correlation_search(graph, 
                                                    weight_threshold=0.0, # threshold for splitting edges
                                                    s=20, max_attempts=2000, max_iters=50000,
                                                    initial=classes)
        
        # store cluster labels
        labels = list()
        for k, c in enumerate(classes):
            for idx in c:
                labels.append(dict(identifier=idx.split('###')[0], cluster=k))
        labels = pd.DataFrame(labels).sort_values('identifier')
        clusters_labels.append(classes)
        
        # compute cluster metrics
        gold_wsi_dfs[i] = gold_wsi_dfs[i].sort_values('identifier')
        
        clusters_metrics['adjusted_rand_score'].append(adjusted_rand_score(labels.cluster.values, gold_wsi_dfs[i].cluster.values))
        clusters_metrics['rand_score'].append(rand_score(labels.cluster.values, gold_wsi_dfs[i].cluster.values))
        clusters_metrics['normalized_mutual_info'].append(normalized_mutual_info_score(labels.cluster.values, gold_wsi_dfs[i].cluster.values))
        clusters_metrics['adjusted_mutual_info'].append(adjusted_mutual_info_score(labels.cluster.values, gold_wsi_dfs[i].cluster.values))
        clusters_metrics['purity'].append(purity_score(labels.cluster.values, gold_wsi_dfs[i].cluster.values))
        clusters_metrics['homogeneity'].append(homogeneity_score(labels.cluster.values, gold_wsi_dfs[i].cluster.values))
        clusters_metrics['fowlkes_mallows'].append(fowlkes_mallows_score(labels.cluster.values, gold_wsi_dfs[i].cluster.values))
        
        # compute cluster distributions
        count = defaultdict(lambda: defaultdict(int))
        for j, cluster in enumerate(classes):
            for node in cluster:
                time_period = int(node.split('###')[-1])
                count[time_period][j]+=1

        prob = [[], []]
        for j in range(len(classes)):
            for t, time_period in enumerate(list(count.keys())):
                if j in count[time_period]:
                    prob[t].append(count[time_period][j] / sum(count[time_period].values()))
                else:
                    prob[t].append(0.)

        clusters_dists.append(prob)
    
    clusters_metrics = {m: np.mean(v) for m, v in clusters_metrics.items()}
        
    return clusters_metrics, clusters_labels, clusters_dists

def LSC(dists, gold_lsc_df):
    y = np.array([jensenshannon(d[0], d[1]) for d in dists])
    y_true = gold_lsc_df.change_graded.values
    return spearmanr(y_true, y)

def COMPARE(dfs, gold_lsc_df):
    y = np.array([np.mean(df.judgment.values) for df in dfs])
    y_true = gold_lsc_df.change_graded.values
    return spearmanr(y_true, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Model evaluation',
        description='Evaluate model on WiC - WSI - LSC')
    parser.add_argument('-o', '--output_file', type=str, default="models-evaluation.tsv")
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-l', '--layer', type=str, default=None)
    parser.add_argument('-b', '--benchmark', type=str)
    parser.add_argument('--wsi_lsc', action='store_true')
    args = parser.parse_args()


    # parameters
    output_file = args.output_file
    benchmark = args.benchmark
    model = args.model
    layer = args.layer
    wsi_lsc = args.wsi_lsc

    # file header
    header = 'benchmark\tmodel\tlayer\ttask\tscore\trows'

    # file output
    if not Path(output_file).is_file():
        lines = [header+ '\n']
    else:
        lines = open(output_file, mode='r', encoding='utf-8').readlines()

    if model == 'xl-lexeme':
        layer = 'tuned'

    if model == 'chatgpt':
        layer = 'chatgpt'
        dfs, gold_wic_dfs, gold_wsi_dfs, gold_lsc_df = read_data_chatgpt(benchmark, wsi=wsi_lsc)
    else:
        dfs, gold_wic_dfs, gold_wsi_dfs, gold_lsc_df = read_data(benchmark, model, layer, wsi=wsi_lsc)

    wic_spearman = WiC(dfs, gold_wic_dfs)
    record = "\t".join([str(i) for i in [benchmark, model, layer, 'wic', wic_spearman[0].round(3), pd.concat(dfs).shape[0]]]) + '\n'
    lines.append(record)
    print(record)

    if wsi_lsc:
        clusters_metrics, clusters_labels, clusters_dists = WSI(dfs, gold_wsi_dfs)

        Path(f'probs/{benchmark}/{model}').mkdir(exist_ok=True, parents=True)
        Path(f'classes/{benchmark}/{model}').mkdir(exist_ok=True, parents=True)
        with open(f'probs/{benchmark}/{model}/{layer}.dill', mode='+wb') as f:
            dill.dump(clusters_dists, f)
            
        with open(f'classes/{benchmark}/{model}/{layer}.dill', mode='+wb') as f:
            dill.dump(clusters_labels, f)
    

        for metric in clusters_metrics:
            record = "\t".join([str(i) for i in [benchmark, model, layer, f'wsi-{metric}', clusters_metrics[metric].round(3), pd.concat(dfs).shape[0]]]) + '\n'
            lines.append(record)
            print(record)

        lsc_spearman = LSC(clusters_dists, gold_lsc_df)
        record = "\t".join([str(i) for i in [benchmark, model, layer, 'lsc', lsc_spearman[0].round(3), pd.concat(dfs).shape[0]]]) + '\n'
        lines.append(record)
        print(record)
    else:        
        lsc_spearman = COMPARE(dfs, gold_lsc_df)
        record = "\t".join([str(i) for i in [benchmark, model, layer, 'lsc', lsc_spearman[0].round(3), pd.concat(dfs).shape[0]]]) + '\n'
        lines.append(record)
        print(record)
        


    with open(output_file, mode='w', encoding='utf-8') as f:
        f.writelines(lines)


    
