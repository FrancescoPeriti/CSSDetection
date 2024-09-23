# A Systematic Comparison of Contextualized Word Embeddings for Lexical Semantic Change

This is the official repository for our paper _A Systematic Comparison of Contextualized Word Embeddings for Lexical Semantic Change_

## Table of Contents

- [Abstract](#abstract)
- [Getting Started](#getting-started)
- [Reproducing Results](#reproducing-results)
- [Evaluation Results](#evaluation-results)
- [References](#references)

## Abstract
Contextualized embeddings are the preferred tool for modeling Lexical Semantic Change (LSC). Current evaluations typically focus on a specific task known as Graded Change Detection (GCD). However, performance comparison across work are often misleading due to their reliance on diverse settings. In this paper, we evaluate state-of-the-art models and approaches for GCD under equal conditions. We further break the LSC problem into Word-in-Context (WiC) and Word Sense Induction (WSI) tasks, and compare models across these different levels. Our evaluation is performed across different languages on eight available benchmarks for LSC, and shows that (i) APD outperforms other approaches for GCD; (ii) XL-LEXEME outperforms other contextualized models for WiC, WSI, and GCD, while being comparable to GPT-4; (iii) there is a clear need for improving the modeling of word meanings, as well as focus on _how_, _when_, and _why_ these meanings change, rather than solely focusing on the extent of semantic change.

## Getting Started
Before you begin, ensure you have met the following requirements:

- Python 3.10.4
- Required Python packages (listed in `requirements.txt`)

To install the required packages, you can use pip:

```bash
pip install -r requirements.txt
```
## Reproducing Results
Reproducing all experiments at once is possible using our comprehensive script, ```run_experiments.sh```. However, please be aware that this may consume a significant amount of time and could be interrupted if your system lacks sufficient GPU resources or if you don't have an OpenAI account. Therefore, we highly recommend executing each command line by line to maintain full control over the experiments.

You find our code in the ```src``` folder. Feel free to contact us if you face any issues!

As GPT-4 is currently an expensive model, we provide our data for the English benchmark in the ```chatgpt_annotation``` directory. Please note that the data is in a raw format suitable for testing our experiments and making comparisons. If you intend to use the data, you might need to process it accordingly.

For those interested in annotating data in languages other than English, input your OpenAI token into the ```chatgpt_api.sh``` file.

If you choose to leverage our data, please remember to mention our work :)

## Evaluation Results
We share our evaluation result in folder ```stats```. The folder contains three file, namely ```layer-stats.tsv```, ```layer-combination-stats.tsv```, and ```wic-wsi-gcd-stats.tsv```.

- ```layer-stats.tsv``` contains the results of Table 1 and 4 of our paper.
- ```layer-combination-stats.tsv``` contains the results of Figure 2 and 3 of our paper.
- ```wic-wsi-gcd-stats.tsv``` contains the results of Table 2 of our paper.

## References
```
@InProceedings{periti-tahmasebi-2024-systematic,
    title = {{A Systematic Comparison of Contextualized Word Embeddings for Lexical Semantic Change}},
    author = "Periti, Francesco  and Tahmasebi, Nina",
    editor = "Duh, Kevin  and Gomez, Helena  and Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.240",
    doi = "10.18653/v1/2024.naacl-long.240",
    pages = "4262--4282",
}
```

```
@Article{periti2024survey,
    author = {Periti, Francesco and Montanelli, Stefano},
    title = {{Lexical Semantic Change through Large Language Models: a Survey}},
    year = {2024},
    issue_date = {November 2024},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {56},
    number = {11},
    issn = {0360-0300},
    url = {https://doi.org/10.1145/3672393},
    doi = {10.1145/3672393},
    journal = {ACM Comput. Surv.},
    month = {jun},
    articleno = {282},
    numpages = {38},
    keywords = {Lexical semantics, lexical semantic change, semantic shift detection, large language models}
}
```

```
@InBook{tahmasebi2021survey,
    author = {Tahmasebi, Nina  and Borin, Lars and Jatowt, Adam},
    title = {{Survey of Computational Approaches to Lexical Semantic Change Detection}},
    booktitle = {Computational approaches to semantic change},
    pages = {1--91},
    publisher = {Language Science Press},
    year = {2021},
    address = {Berlin},
    doi = {10.5281/zenodo.5040302}
}
```
