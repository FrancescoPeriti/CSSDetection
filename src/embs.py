import os
import csv
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from utils import XLexeme, BERTlikeModel

# process corpora
def processing(benchmark, uses, target):
    if 'dwug_ru12' in benchmark:
        uses['grouping'] = [1 if '1700-1916' == i else 2 for i in uses['grouping']]
    elif 'dwug_ru23' in benchmark:
        uses['grouping'] = [1 if '1918-1990' == i else 2 for i in uses['grouping']]
    elif 'dwug_ru13' in benchmark:
        uses['grouping'] = [1 if '1700-1916' == i else 2 for i in uses['grouping']]
    elif 'dwug_no12' in benchmark:
        uses['grouping'] = [1 if '1970-2015' == i else 2 for i in uses['grouping']]
    elif 'dwug_no23' in benchmark:
        uses['grouping'] = [1 if '1980-1990' == i else 2 for i in uses['grouping']]

    if 'dwug_ru' in benchmark:
        uses['identifier'] = [str(i).split('_')[-1] for i in uses['identifier']]
    
    try:
        clusters = pd.read_csv(f'{benchmark}/clusters/opt/{target}.csv', sep='\t')
        uses_to_remove = clusters[clusters['cluster']==-1].identifier.values
        uses = uses[~uses.identifier.isin(uses_to_remove)]
    except: 
        pass
    
    return uses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Embedding extraction',
        description='Extract embeddings for word usages in C1 and C2')
    parser.add_argument('--benchmark', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--subword_prefix', type=str, choices=['##', '_', 'Ġ'], default='##', help='special token used for subwords tokenization')
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


    # read corpora
    corpora = dict()
    for target in os.listdir(f'{benchmark}/data'):
        # read uses
        uses = pd.read_csv(f'{benchmark}/data/{target}/uses.csv', sep='\t', quoting=csv.QUOTE_NONE)
    
        # create corpora
        uses = processing(benchmark, uses, target)
        C1 = uses[uses['grouping']==1]
        C2 = uses[uses['grouping']==2]
        
        # remove useless columns
        C1 = C1[['identifier', 'context', 'indexes_target_token', 'grouping']]
        C2 = C2[['identifier', 'context', 'indexes_target_token', 'grouping']]
        
        # store corpora
        corpora[target] = (C1, C2)

    
    if pretrained_model == 'pierluigic/xl-lexeme':
        model = XLexeme(pretrained_model, device)
    else:
        model = BERTlikeModel(pretrained_model, device, batch_size=batch_size, subword_prefix=subword_prefix, max_length=max_length)

    # embeddings extraction
    for target in corpora:
        E_1, E_2 = model.encode(corpora[target][0]), model.encode(corpora[target][1])
        
        if isinstance(model, XLexeme):
            Path(f'{benchmark}/embeddings/{model_name}/corpus1/tuned/').mkdir(exist_ok=True, parents=True)
            Path(f'{benchmark}/embeddings/{model_name}/corpus2/tuned/').mkdir(exist_ok=True, parents=True)
            np.save(f'{benchmark}/embeddings/{model_name}/corpus1/tuned/{target}', E_2)
            np.save(f'{benchmark}/embeddings/{model_name}/corpus2/tuned/{target}', E_1)        
        else:
            n_layers = E_1.shape[0]
            for layer in range(n_layers):
                Path(f'{benchmark}/embeddings/{model_name}/corpus1/{layer+1}/').mkdir(exist_ok=True, parents=True)
                Path(f'{benchmark}/embeddings/{model_name}/corpus2/{layer+1}/').mkdir(exist_ok=True, parents=True)
                np.save(f'{benchmark}/embeddings/{model_name}/corpus1/{layer+1}/{target}', E_1[layer])
                np.save(f'{benchmark}/embeddings/{model_name}/corpus2/{layer+1}/{target}', E_2[layer])
