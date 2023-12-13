import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from collections import defaultdict
import matplotlib.pyplot as plt

def boxplot(benchmarks, filename='tmp.png'):
    measures=['apd', 'prt', 'ap+jsd', 'app+apdp']
    models=['bert', 'mbert', 'xlm-r']

    # read data
    df = pd.read_csv(f'/mimer/NOBACKUP/groups/cik_data/rqlsc/stats.tsv', sep='\t')
    combo_df = pd.read_csv(f'/mimer/NOBACKUP/groups/cik_data/rqlsc/comb_stats.tsv', sep='\t')
    combo_df['spearman'] = np.abs(combo_df['spearman'].values)
    df['spearman'] = np.abs(df['spearman'].values)

    # create figure
    fig, ax = plt.subplots(len(measures), len(benchmarks), figsize=(2.5*len(benchmarks), 2*len(benchmarks)), sharey=True, sharex=True)

    # set y limits
    y_min = min(combo_df.spearman.values.min(), df.spearman.values.min())
    y_max = max(combo_df.spearman.values.max(), df.spearman.values.max())

    for i, measure in enumerate(measures):
        for j, benchmark in enumerate(benchmarks):
            scores = defaultdict(list)
            all_scores = defaultdict(list)
            
            for k, model in enumerate(models):
                if model == 'bert' and benchmark == 'dwug_la12':
                    all_scores[model] = list()
                    continue
                    
                # combinations
                combo_sub_df = combo_df[(combo_df['measure'] == measure) & 
                                        (combo_df['model'] == model) & 
                                        (combo_df['benchmark'] == benchmark)].reset_index(drop=True)
                all_scores[model].extend(combo_sub_df.spearman.values.tolist())
                
                # sum_8-9-10-11
                scores[k+1].append(combo_sub_df[combo_sub_df['comb']=='sum_8-9-10-11'].spearman.values[0])
                
                # single layers
                sub_df = df[(df['measure'] == measure) & 
                            (df['model'] == model) & 
                            (df['benchmark'] == benchmark)]
                sub_df['layer'] = sub_df['layer'].astype(int)
                all_scores[model].extend(sub_df.spearman.values.tolist())
                
                # top layer
                scores[k+1].append(sub_df.sort_values('spearman', ascending=False).spearman.values[0])
                
                # last layer
                scores[k+1].append(sub_df.sort_values('layer', ascending=False).spearman.values[0])
                
                # 8th layer
                scores[k+1].append(sub_df.sort_values('layer', ascending=True).spearman.values[8])
            
            ax[i][j].boxplot(all_scores.values())
            ax[i][j].set_ylim(y_min-0.01, 0.79)
            ax[i][j].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            
            for model_x in scores:
                if i==0 and j==0 and model_x == 1:
                    ax[i][j].scatter(model_x+0.3, scores[model_x][0], color='#D81B60', marker='*', alpha=0.5, label='⨁ Layers 9-12')
                    #ax[i][j].scatter(model_x+0.2, scores[model_x][1], color='#1E88E5', marker='o', alpha=0.5)
                    ax[i][j].scatter(model_x+0.3, scores[model_x][2], color='#FFC107', marker='o', alpha=0.5, label='Layer 12')
                    ax[i][j].scatter(model_x+0.3, scores[model_x][3], color='#004D40', marker='o', alpha=0.5, label='Layer 8')
                
                else:
                    ax[i][j].scatter(model_x+0.3, scores[model_x][0], color='#D81B60', marker='*', alpha=0.7)
                    #ax[i][j].scatter(model_x+0.2, scores[model_x][1], color='#1E88E5', marker='o', alpha=0.5)
                    ax[i][j].scatter(model_x+0.3, scores[model_x][2], color='#FFC107', marker='o', alpha=0.7)
                    ax[i][j].scatter(model_x+0.3, scores[model_x][3], color='#004D40', marker='o', alpha=0.7)
                
            #if i == 0 and j == 0:
            #    ax[i][j].legend()
            
            if i == 0:
                title = f'{benchmark}'.replace('dwug_', '').upper()
                if not ('RU' in title or 'NO' in title):
                    title = title.replace('12', '')
                ax[i][j].set_title(title)
            
            if j == 0:
                ax[i][j].set_ylabel(f"{measure.upper().replace('APP+APDP', 'WiDiD')}\nSpearman\'s correlation")

            
            ax[i][j].set_xticks([1, 2, 3])
            ax[i][j].set_xticklabels(['BERT', 'mBERT', 'XLM-R'])
            
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches = "tight")


benchmarks = ['dwug_en12', 'dwug_la12', 'dwug_de12', 'dwug_sv12', 'dwug_es12']
boxplot(benchmarks, filename='en_la_de_sv_es.png')

benchmarks = ['dwug_ru12', 'dwug_ru23', 'dwug_ru13', 'dwug_no12', 'dwug_no23', 'dwug_zh12']
boxplot(benchmarks, filename='ru_no_zh.png')