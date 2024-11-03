import random
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
import string
import time
from datetime import datetime
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, GPT2Model,
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel
)

# Pre-trained models initialization
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2Model.from_pretrained('gpt2')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
xlm_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
xlm_model = RobertaModel.from_pretrained('roberta-base')

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model = gpt2_model.to(device)
bert_model = bert_model.to(device)
xlm_model = xlm_model.to(device)

def gpt2_encoder(s, no_wordpiece=0):
    """Compute semantic vectors with GPT2"""
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in gpt2_tokenizer.get_vocab().keys()]
        s = " ".join(words)
    try:
        inputs = gpt2_tokenizer(s, return_tensors='pt', max_length=512, padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = gpt2_model(**inputs)
        v = torch.mean(outputs.last_hidden_state, 1)
        return v[0].cpu().numpy()
    except Exception as _:
        return np.zeros((768,))

def bert_encoder(s, no_wordpiece=0):
    """Compute semantic vector with BERT"""
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = bert_tokenizer(s, return_tensors='pt', max_length=512, padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    v = torch.mean(outputs.last_hidden_state, 1)
    return v[0].cpu().numpy()

def xlm_encoder(s, no_wordpiece=0):
    """Compute semantic vector with XLM"""
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in xlm_tokenizer.get_vocab().keys()]
        s = " ".join(words)
    inputs = xlm_tokenizer(s, return_tensors='pt', max_length=512, padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = xlm_model(**inputs)
    v = torch.mean(outputs.last_hidden_state, 1)
    return v[0].cpu().numpy()

def clean(s):
    """Preprocess log message"""
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    return " ".join([word.lower().strip() for word in content.strip().split()])

class LogDataset(Dataset):
    """PyTorch Dataset for log sequences"""
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return sequence, label
        return sequence

def create_dataloader(x, y=None, batch_size=32, shuffle=True):
    """Create PyTorch DataLoader"""
    dataset = LogDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_HDFS(log_file, label_file=None, train_ratio=0.5, window='session',
              split_type='uniform', e_type="bert", no_word_piece=0, batch_size=32):
    """Load HDFS dataset and return PyTorch DataLoaders"""
    print('====== Input data summary ======')

    e_type = e_type.lower()
    if e_type == "bert":
        encoder = bert_encoder
    elif e_type == "xlm":
        encoder = xlm_encoder
    elif e_type == "gpt2":
        encoder = gpt2_encoder
    else:
        raise ValueError('Embedding type must be bert, xlm, or gpt2')

    t0 = time.time()
    assert log_file.endswith('.log'), "Missing .log file"
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    
    # Load and process logs
    print(f"Loading {log_file}")
    with open(log_file, 'r', encoding='utf8') as f:
        logs = [x.strip() for x in f.readlines()]
    
    data_dict = OrderedDict()
    E = {}
    
    # Process logs
    for i, line in enumerate(logs):
        timestamp = " ".join(line.split()[:2])
        timestamp = datetime.strptime(timestamp, '%y%m%d %H%M%S').timestamp()
        blkId_list = re.findall(r'(blk_-?\d+)', line)
        blkId_list = list(set(blkId_list))
        if len(blkId_list) >= 2:
            continue
            
        content = clean(line).lower()
        if content not in E:
            E[content] = encoder(content, no_word_piece)
            
        for blk_Id in set(blkId_list):
            if blk_Id not in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append((E[content], timestamp))
            
        if (i + 1) % 1000 == 0:
            print(f"\rProcessing {(i+1)/len(logs)*100:.2f}% - {len(E)} unique messages", end="")
            
    # Process sequences and timestamps
    data_df = []
    for k, v in data_dict.items():
        seq = [x[0] for x in v]
        rt = [x[1] for x in v]
        rt = [rt[i] - rt[i-1] for i in range(1, len(rt))]
        rt = [0] + rt
        data_df.append((k, seq, rt))
    
    data_df = pd.DataFrame(data_df, columns=['BlockId', 'EventSequence', 'TimeSequence'])

    if label_file:
        # Process labels
        label_data = pd.read_csv(label_file)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        
        # Split data
        x_data = np.array(data_df['EventSequence'].values.tolist())
        y_data = data_df['Label'].values
        
        # Create train/test split
        indices = np.random.permutation(len(x_data))
        train_size = int(train_ratio * len(x_data))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        x_train, y_train = x_data[train_indices], y_data[train_indices]
        x_test, y_test = x_data[test_indices], y_data[test_indices]
        
        # Create DataLoaders
        train_loader = create_dataloader(x_train, y_train, batch_size=batch_size)
        test_loader = create_dataloader(x_test, y_test, batch_size=batch_size)
        
        return train_loader, test_loader, (x_train, y_train), (x_test, y_test)
    
    raise NotImplementedError("Missing label file for the HDFS dataset!")

def load_supercomputers(log_file, train_ratio=0.5, windows_size=20, step_size=0, 
                       e_type='bert', mode="balance", no_word_piece=0, batch_size=32):
    """Load supercomputer logs and return PyTorch DataLoaders"""
    print(f"Loading {log_file}")
    
    with open(log_file, 'r', encoding='utf8') as f:
        logs = [x.strip() for x in f.readlines()]
        
    e_type = e_type.lower()
    if e_type == "bert":
        encoder = bert_encoder
    elif e_type == "xlm":
        encoder = xlm_encoder
    elif e_type == "gpt2":
        encoder = gpt2_encoder
    else:
        raise ValueError('Embedding type must be bert, xlm, or gpt2')
        
    E = {}
    x_tr, y_tr = [], []
    n_train = int(len(logs) * train_ratio)
    
    # Process training data
    i = 0
    while i < n_train - windows_size:
        seq = []
        label = 0
        
        for j in range(i, i + windows_size):
            if logs[j][0] != "-":
                label = 1
            content = logs[j][logs[j].find(' ')+1:]
            content = clean(content.lower())
            
            if content not in E:
                E[content] = encoder(content, no_word_piece)
            seq.append(E[content])
            
        x_tr.append(seq)
        y_tr.append(label)
        i += step_size if step_size > 0 else windows_size
        
        if len(x_tr) % 1000 == 0:
            print(f"\rProcessing training data: {i/n_train*100:.2f}%", end="")
            
    # Process test data
    x_te, y_te = [], []
    for i in range(n_train, len(logs) - windows_size, step_size if step_size > 0 else windows_size):
        seq = []
        label = 0
        
        for j in range(i, i + windows_size):
            if logs[j][0] != "-":
                label = 1
            content = logs[j][logs[j].find(' ')+1:]
            content = clean(content.lower())
            
            if content not in E:
                E[content] = encoder(content, no_word_piece)
            seq.append(E[content])
            
        x_te.append(seq)
        y_te.append(label)
        
        if len(x_te) % 1000 == 0:
            print(f"\rProcessing test data: {(i-n_train)/(len(logs)-n_train)*100:.2f}%", end="")
            
    if mode == 'balance':
        x_tr, y_tr = balancing(x_tr, y_tr)
        
    # Create DataLoaders
    train_loader = create_dataloader(x_tr, y_tr, batch_size=batch_size)
    test_loader = create_dataloader(x_te, y_te, batch_size=batch_size)
    
    return train_loader, test_loader, (x_tr, y_tr), (x_te, y_te)