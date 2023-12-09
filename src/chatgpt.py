import os
import csv
import time
import openai
openai.api_key_path = 'chatgpt_api.txt'
import random
import argparse
import numpy as np
import pandas as pd
from embs import processing
from stats import gold_scores, permutation_test
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm

from openai import OpenAI
client = OpenAI(api_key=open('chatgpt_api.txt').read()) 

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

random.seed(42)

def wait_random_time(min_=60, max_=75):
    # There is no hourly usage limit for ChatGPT,
    # but each response is subject to a word and
    # character limit of approximately 500
    # words or 4000 characters
    return random.choice(list(range(min_, max_)))

# You have to send the entire conversation back to the API each time.
# ChatGPT does not remember conversations, it is just sending what is
# in the chat window back to the API every time you hit submit.
def input_prompt(messages, temperature, max_tokens):
    res = completion_with_backoff(model=model, temperature=temperature, messages=messages, max_tokens=max_tokens)
    return res.choices[0].message.content
    #chatgpt = openai.ChatCompletion.create(model=model, temperature=temperature, messages=messages, max_tokens=max_tokens, request_timeout=120)
    #return chatgpt['choices'][0]['message']['content']

parser = argparse.ArgumentParser(
                    prog='Embedding extraction',
                    description='Extract embeddings for word usages in C1 and C2')
parser.add_argument('--benchmark', type=str)
args = parser.parse_args()

# parameters
benchmark=args.benchmark

# chatgpt parameters
# https://arxiv.org/pdf/2307.11760.pdf
start_content = "Determine whether an input word has the same meaning in the two input sentences. Answer with 'Same', 'Related', 'Linked', or 'Distinct'. This is very important to my career."
model = 'gpt-4'
temperature = 0
max_tokens=4
model_name='chatgpt'
layer=str(temperature)

# target words
targets = sorted(os.listdir(f'{benchmark}/data/'))

# Dataframe per target
dfs = list()
scores = dict()
for target in targets:
    Path(f'chatgpt_annotation/{benchmark}/').mkdir(exist_ok=True, parents=True)
    if Path(f'chatgpt_annotation/{benchmark}/{target}.npy').is_file(): continue
    
    # uses
    uses = pd.read_csv(f'{benchmark}/data/{target}/uses.csv', sep='\t', quoting=csv.QUOTE_NONE)

    # judgments
    judgments = pd.read_csv(f'{benchmark}/data/{target}/judgments.csv', sep='\t')

    # remove useless columns
    uses = processing(benchmark, uses, targets)
    uses = uses[['identifier', 'context', 'indexes_target_token', 'grouping']]
    judgments = judgments[['identifier1', 'identifier2', 'annotator', 'judgment']]

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
    df1 = df[['identifier1', 'context1', 'grouping1', 'indexes_target_token1']]
    df1 = df1.rename(columns={c: c[:-1] for c in df1.columns})
    df2 = df[['identifier2', 'context2', 'grouping2', 'indexes_target_token2']]
    df2 = df2.rename(columns={c: c[:-1] for c in df2.columns})

    first_message = {"role": "system", "content": start_content}

    tmp_scores = list()
    for k, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Annotating {target}'):
        prompt = f'''Determine whether "{target.replace('_nn', '').replace('_vb', '')}" has the same meaning in the following sentences. Do they refer to roughly the Same, different but closely Related, distant/figuratively Linked or unrelated Distinct word meanings?
            1. Sentence 1: {row["context1"]}
            2. Sentence 2: {row["context2"]}
            '''

        messages = [first_message, {"role": "user", "content": prompt}]

        content_ = input_prompt(messages, temperature, max_tokens)
        try:
            content_ = int(content_.replace('Same', '4').replace('Related', '3').replace('Linked', '2').replace('Distinct', '1').replace("'", ''))
        except:
            # if the model fail in answering following the template, we assign 2.5 (midpoint in the DURel scale).
            content_=2.5

        tmp_scores.append(content_)

    scores[target] = np.array(tmp_scores)
    np.save(f'chatgpt_annotation/{benchmark}/{target}', scores[target])

# load ground truth
y_true = gold_scores(benchmark)
header = f'benchmark\nmodel\tlayer\tspearman\tspearmanp'

# file output
stats_file = "chatgpt_stats.tsv"
if not Path(stats_file).is_file():
    lines = [header + '\n']
else:
    lines = open(stats_file, mode='r', encoding='utf-8').readlines()

y = np.array([scores[t].mean() for t in targets])
sp, pt = spearmanr(y_true, y), permutation_test(y_true, y)
lines.append(f'{benchmark}\t{model_name}\t{layer}\t{round(sp.correlation, 3)}\t{round(sp.pvalue, 3)}\n')

with open(stats_file, mode='w', encoding='utf-8') as f:
    f.writelines(lines)
