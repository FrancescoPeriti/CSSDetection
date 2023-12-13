import os
import dill
import csv
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cosine
from correlation_clustering import cluster_correlation_search
from utils import XLexeme, BERTlikeModel
from embs import processing
from stats import gold_scores, permutation_test
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from collections import defaultdict

parser = argparse.ArgumentParser(
                    prog='Embedding extraction',
                    description='Extract embeddings for word usages in C1 and C2')
parser.add_argument('--benchmark', type=str)
parser.add_argument('--layer', type=int, default=12)
parser.add_argument('--model_name', type=str)
parser.add_argument('--pretrained_model', type=str)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--subword_prefix', type=str, choices=['##', '_'], default='##', help='special token used for subwords tokenization')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_length', type=int, default=512)
args = parser.parse_args()

# parameters
benchmark=args.benchmark
model_name=args.model_name
pretrained_model=args.pretrained_model
batch_size=args.batch_size
subword_prefix=args.subword_prefix
max_length=args.max_length
device=args.device
layer=args.layer-1

# target words
targets = sorted(os.listdir(f'{benchmark}/data/'))

# Embeddings for sentences <s_1, s_2>
E = dict()

# load embedding model
if pretrained_model == 'pierluigic/xl-lexeme':
    model = XLexeme(pretrained_model, device)
    layer='tuned'
else:
    model = BERTlikeModel(pretrained_model, device, batch_size=batch_size, subword_prefix=subword_prefix,
                          max_length=max_length)

# Dataframe per target
dfs = list()
scores = defaultdict(list)
for target in targets:
    # uses
    uses = pd.read_csv(f'{benchmark}/data/{target}/uses.csv', sep='\t', quoting=csv.QUOTE_NONE)

    # judgments
    if 'dwug_no' in benchmark:
        judgments = pd.read_csv(f'{benchmark}/data_joint/data_joint.tsv', sep='\t')
        judgments = judgments[judgments['lemma']==target]
    else:
        judgments = pd.read_csv(f'{benchmark}/data/{target}/judgments.csv', sep='\t')

    # remove useless columns
    uses = processing(benchmark, uses, targets)
    uses = uses[['identifier', 'context', 'indexes_target_token', 'grouping']]
    judgments = judgments[['identifier1', 'identifier2', 'annotator', 'judgment']]

    uses['identifier'] = uses['identifier'].astype(str)
    judgments['identifier1'] = judgments['identifier1'].astype(str)
    judgments['identifier2'] = judgments['identifier2'].astype(str)
    
    # join the two dataframes
    df = judgments.merge(uses, left_on='identifier1', right_on='identifier')
    df = df.rename(
        columns={'context': 'context1', 'indexes_target_token': 'indexes_target_token1', 'grouping': 'grouping1'})
    del df['identifier']
    df = df.merge(uses, left_on='identifier2', right_on='identifier')
    df = df.rename(
        columns={'context': 'context2', 'indexes_target_token': 'indexes_target_token2', 'grouping': 'grouping2'})
    del df['identifier']

    # take the mean of annotators
    df = df.groupby(
        ['identifier1', 'identifier2', 'context1', 'context2', 'indexes_target_token1', 'indexes_target_token2',
         'grouping1', 'grouping2']).mean().reset_index()
    dfs.append(df)
    examples = list()
    
    # split df
    df1 = df[['identifier1', 'context1', 'grouping1', 'indexes_target_token1']].reset_index(drop=True)
    df1 = df1.rename(columns={c: c[:-1] for c in df1.columns})
    df2 = df[['identifier2', 'context2', 'grouping2', 'indexes_target_token2']].reset_index(drop=True)
    df2 = df2.rename(columns={c: c[:-1] for c in df2.columns})

    # embeddings extraction (this is ideal, but some sentences may be too long)
    #E_1, E_2 = model.encode(df1), model.encode(df2)
    #if isinstance(model, XLexeme):
    #    E[target] = (E_1, E_2)
    #else:
    #    E[target] = (E_1[layer], E_2[layer])


    if not isinstance(model, XLexeme):
        # some sentences are too long. We exclude them
        idx1, idx2 = model.filter_safe_idx(df1), model.filter_safe_idx(df2)
        idx = np.intersect1d(idx1, idx2)

        # all the pairs contain a long sentence
        if idx.shape[0] == 0:
            continue

        df1, df2 = df1.loc[idx], df2.loc[idx]
        E1, E2 = model.encode(df1), model.encode(df2)
        E[target] = (E1[layer], E2[layer])

        for i, idx in enumerate(zip(df1.identifier.values, df2.identifier.values)):
            id1, id2 = idx
            scores[target].append(dict(identifier1=id1, identifier2=id2, judgment=1-cosine(E1[layer][i], E2[layer][i])))
    else:
        E1, E2 = list(), list()
        for j, row in df1.iterrows():
            row1, row2 = pd.DataFrame([dict(row)]), pd.DataFrame([dict(df2.loc[j])])

            e_1, e_2 = model.encode(row1), model.encode(row2)
            try:
                e_1, e_2 = model.encode(row1), model.encode(row2)
            except:
                # one sentence is too long
                continue
            else:
                #if isinstance(model, XLexeme):
                E1.append(e_1[0])
                E2.append(e_2[0])
                #else:
                #    E1.append(e_1[layer][0])
                #    E2.append(e_2[layer][0])
                scores[target].append(dict(identifier1=row1['identifier'], identifier2=row2['identifier'], judgment=1-cosine(E1[-1], E2[-1])))

        try:
            E[target] = (np.array(E1), np.array(E2))
        except:
            continue


Path(f'scores/{benchmark}/{model_name}').mkdir(exist_ok=True, parents=True)
with open(f'scores/{benchmark}/{model_name}/{layer if layer == "tuned" else str(int(layer)+1)}.dill', mode='+wb') as f:
    dill.dump(scores, f)
