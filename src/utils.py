from datasets import Dataset
import re
import warnings
import pandas as pd
import numpy as np
import torch
import random
from WordTransformer import WordTransformer, InputExample
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging
from datasets.utils import logging as datasets_logging

# avoid boring logging
transformers_logging.set_verbosity_error()
datasets_logging.set_verbosity_error()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
SEED = 42
set_seed(SEED)


class XLexeme:
    def __init__(self, model:str='pierluigic/xl-lexeme', device:str='cuda'):
        self._model = WordTransformer(model, device=device)

    def encode(self, df: pd.DataFrame) -> np.array:
        examples = list()
        for _, row in df.iterrows():
            start, end = row["indexes_target_token"].split(':')
            start, end = int(start), int(end)
            examples.append(InputExample(texts=row[f"context"], positions=[start, end]))

        return self._model.encode(examples)



class BERTlikeModel:
    def __init__(self,
                 model: str = 'bert-base-uncased', device:str='cuda',
                 subword_prefix: str = '##',
                 batch_size:int=16,
                 max_length:int=512):

        self._set_device(device)

        # load hugginface tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, output_hidden_states=True)

        # some attributes
        self.pretrained = model  # model name
        self.subword_prefix = subword_prefix  # subword prefix
        self.num_layers = self.model.config.num_hidden_layers  # number of model's layers
        self.num_heads = self.model.config.num_attention_heads  # number of model's heads
        self.batch_size = batch_size # batch size
        self.max_length = max_length # maximum number of input tokens

        # load model on gpu/cpu memory
        _ = self.model.to(self._device)

        # evaluation mode
        _ = self.model.eval()

    def _set_device(self, device: str = 'cuda') -> None:
        if device == 'cuda' and torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"

        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("No GPU found. CPU will be used", category=UserWarning)

        self._device = torch.device(device_name)

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        def tokenize(examples) -> dict:
            """Tokenization function"""
            return self.tokenizer(examples["context"],
                                  return_tensors='pt',
                                  padding="max_length",
                                  max_length=self.max_length if self.max_length else self.tokenizer.model_max_length,
                                  truncation=True).to(self._device)

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format('torch')
        return dataset

    def _load_dataset(self, df: pd.DataFrame) -> Dataset:
        records = list()
        for _, row in df.iterrows():
            row = row.to_dict()
            start, end = row["indexes_target_token"].split(':')
            row['start']=int(start)
            row['end']=int(end)
            records.append(row)
        return Dataset.from_list(records)


    def filter_safe_idx(self, df:pd.DataFrame) -> np.array:
        # load dataset
        dataset = self._load_dataset(df)

        # split text from other data
        text = dataset.select_columns('context')
        offset = dataset.remove_columns('context')

        # tokenize text
        tokenized_text = self._tokenize_dataset(text)

        idx = list()
        index=-1
        for i in range(0, tokenized_text.shape[0], self.batch_size):
            start, end = i, min(i + self.batch_size, text.num_rows)
            batch_offset = offset.select(range(start, end))
            batch_text = text.select(range(start, end))
            batch_tokenized_text = tokenized_text.select(range(start, end))

            # select the embeddings of a specific target word
            for j, row in enumerate(batch_tokenized_text):
                index+=1
                # string containing tokens of the j-th sequence
                input_tokens = row['input_ids'].tolist()
                input_tokens_str = " ".join(self.tokenizer.convert_ids_to_tokens(input_tokens))

                # string containing tokens of the target word occurrence
                word_tokens = batch_text[j]['context'][batch_offset[j]['start']:batch_offset[j]['end']]
                word_tokens_str = " ".join(self.tokenizer.tokenize(word_tokens))

                try:
                    if word_tokens_str.startswith('▁ '):
                        word_tokens_str = word_tokens_str[2:]
                    if word_tokens_str[0] == ' ':
                        word_tokens_str = word_tokens_str[1:]
                except:
                    print(word_tokens, ' - ', batch_text[j]['context'])
                    continue

                # search the occurrence of 'word_tokens_str' in 'input_tokens_str' to get the corresponding position
                try:
                    pos_offset, pos_error, pos = 0, None, None
                    while True:
                        tmp = input_tokens_str[pos_offset:]
                        match = re.search(f"( +|^){re.escape(word_tokens_str)}(?!\w+\b)", tmp, re.DOTALL)
                        # match = re.search(f"( +|^){word_tokens_str}(?!\w+| self.subword_prefix)", tmp, re.DOTALL)

                        if match is None and self.pretrained == "ChangeIsKey/roberta-kubhist2":
                            match = re.search(f"( +|^){re.escape(self.subword_prefix + word_tokens_str)}(?!\w+\b)",
                                              tmp,
                                              re.DOTALL)

                        if match is None:
                            break

                        current_pos = pos_offset + match.start()
                        current_error = abs(current_pos - batch_offset[j]['start'])

                        if pos is None or current_error < pos_error:
                            pos = current_pos
                            pos_error = current_error
                        else:
                            break

                        pos_offset += match.end()
                        idx.append(index)
                except:
                    
                    continue

        return np.array([i for i in sorted(set(idx))])
    
    
    def encode(self, df: pd.DataFrame) -> np.array:
        # load dataset
        dataset = self._load_dataset(df)

        # split text from other data
        text = dataset.select_columns('context')
        offset = dataset.remove_columns('context')

        # tokenize text
        tokenized_text = self._tokenize_dataset(text)

        # collect embedding to store on disk
        embeddings = dict()
        for i in range(0, tokenized_text.shape[0], self.batch_size):
            start, end = i, min(i + self.batch_size, text.num_rows)
            batch_offset = offset.select(range(start, end))
            batch_text = text.select(range(start, end))
            batch_tokenized_text = tokenized_text.select(range(start, end))

            model_input = dict()

            # to device
            model_input['input_ids'] = batch_tokenized_text['input_ids'].to(self._device)

            # XLM-R doesn't use 'token_type_ids'
            if 'token_type_ids' in batch_tokenized_text:
                model_input['token_type_ids'] = batch_tokenized_text['token_type_ids'].to(self._device)

            model_input['attention_mask'] = batch_tokenized_text['attention_mask'].to(self._device)

            # model prediction
            with torch.no_grad():
                model_output = self.model(**model_input)

            # hidden states
            hidden_states = torch.stack(model_output['hidden_states'])

            # select the embeddings of a specific target word
            for j, row in enumerate(batch_tokenized_text):
                # string containing tokens of the j-th sequence
                input_tokens = row['input_ids'].tolist()
                input_tokens_str = " ".join(self.tokenizer.convert_ids_to_tokens(input_tokens))

                # string containing tokens of the target word occurrence
                word_tokens = batch_text[j]['context'][batch_offset[j]['start']:batch_offset[j]['end']]
                word_tokens_str = " ".join(self.tokenizer.tokenize(word_tokens))

                if word_tokens_str.startswith('▁ '):
                    word_tokens_str = word_tokens_str[2:]
                if word_tokens_str[0] == ' ':
                    word_tokens_str = word_tokens_str[1:]
                
                # search the occurrence of 'word_tokens_str' in 'input_tokens_str' to get the corresponding position
                try:
                    matches, pos_offset, pos_error, pos = list(), 0, None, None
                    while True:
                        tmp = input_tokens_str[pos_offset:]
                        match = re.search(f"( +|^){re.escape(word_tokens_str)}(?!\w+\b)", tmp, re.DOTALL)
                        #match = re.search(f"( +|^){word_tokens_str}(?!\w+| self.subword_prefix)", tmp, re.DOTALL)

                        if match is None and self.pretrained == "ChangeIsKey/roberta-kubhist2":
                            match = re.search(f"( +|^){re.escape(self.subword_prefix+word_tokens_str)}(?!\w+\b)", tmp, re.DOTALL)
                        
                        if match is None:
                            break

                        current_pos = pos_offset + match.start()
                        current_error = abs(current_pos - batch_offset[j]['start'])

                        if pos is None or current_error < pos_error:
                            pos = current_pos
                            pos_error = current_error
                        else:
                            break

                        pos_offset += match.end()
                        matches.append(match)
                except:
                    print(word_tokens_str, input_tokens_str)
                    idx_original_sent = batch_tokenized_text.num_rows * i + j
                    #warnings.warn(f"An error occurred with the {idx_original_sent}-th sentence: {batch_text[j]}. It will be ignored",
                    #    category=UserWarning)
                    continue

                # Truncation side effect: the target word is over the maximum input length
                if len(matches) == 0:
                    idx_original_sent = batch_tokenized_text.num_rows * i + j
                    #warnings.warn(f"An error occurred with the {idx_original_sent}-th sentence: {batch_text[j]}. It will be ignored",
                    #    category=UserWarning)
                    continue

                n_previous_tokens = len(input_tokens_str[:pos].split())  # number of tokens before that sub-word
                n_word_token = len(word_tokens_str.split())  # number of tokens of the target word

                # Store the embeddings from each layer
                for layer in range(1, self.num_layers + 1):
                    # embeddings of each sub-words
                    sub_word_state = hidden_states[layer, j][n_previous_tokens: n_previous_tokens + n_word_token]

                    # mean of sub-words embeddings
                    word_state = torch.sum(sub_word_state, dim=0).unsqueeze(0) # sum instead of mean, as mean may lead to nan values

                    if layer in embeddings:
                        embeddings[layer] = torch.vstack([embeddings[layer], word_state])
                    else:
                        embeddings[layer] = word_state

            # empty cache
            torch.cuda.empty_cache()

        return torch.cat([embeddings[layer].unsqueeze(0) for layer in range(1, self.num_layers + 1)]).detach().cpu().numpy()
