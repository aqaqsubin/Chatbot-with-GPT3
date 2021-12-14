
import pandas as pd
import torch
import os
import errno
import numpy as np
from ast import literal_eval
from shutil import rmtree

def to_cuda(batch, gpuid):
    device = torch.device('cuda:%d' % gpuid)
    for i, n in enumerate(batch):
        if n != "data":
            batch[n] = batch[n].to(dtype=torch.long, device=device)

def saveCSVFile(filepath, dist):
    dist.to_csv(filepath, mode='w', sep='\t', index=False, encoding='utf-8')

def saveXLSXFile(filepath, dist):
    with pd.ExcelWriter(filepath) as writer:
        dist.to_excel(writer, sheet_name='sheet_name_1',engine='xlsxwriter')

def get_filetype(filepath):
    return filepath.split('/')[-1].split('.')[1]

def read_df(filepath, sep='\t'):
    filetype = get_filetype(filepath)
    if filetype == 'csv':
        data = pd.read_csv(filepath, sep=sep, converters={
            'neg_pools' : literal_eval
        })
    elif filetype == 'xlsx':
        data = pd.read_excel(filepath, engine='openpyxl')
    return data

def tokenize(sent, tokenizer, max_len=128):
    tokens = tokenizer.tokenize(tokenizer.cls_token + str(sent) + tokenizer.sep_token)
    seq_len = len(tokens)
    if seq_len > max_len:
        tokens = tokens[:max_len-1] + [tokens[-1]]
        seq_len = len(tokens)
        assert seq_len == len(tokens), f'{seq_len} ==? {len(tokens)}'
        
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    while len(token_ids) < max_len:
        token_ids += [tokenizer.pad_token_id]

    return token_ids

def encode(sent, tokenizer, max_len):
    tok_ids = tokenize(sent, tokenizer=tokenizer, max_len=max_len)
    return torch.unsqueeze(torch.LongTensor(tok_ids), 0)

def collate_mp(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def del_folder(path):
    try:
        rmtree(path)
    except:
        pass