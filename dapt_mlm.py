# %%
from dataclasses import replace
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datasets, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)
# from transformers import get_scheduler
from datasets import load_dataset, Dataset

from tqdm import tqdm
import numpy as np
from termcolor import colored
from collections import Counter
from munch import Munch

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments
from transformers import Trainer

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Second Masked Language Modelling pretraining on unlabeled dataset. First step for DAPT (domain-adaptive pre-training)")

    parser.add_argument("--dataset", type=str, default="data/dummy_dataset.json",
                        help="Dataset path (default one is a dummy dataset with same format as the original (confidential)"
                            "Default: data/dummy_dataset.json")

    return parser.parse_args()

cmdargs = parse_args()


args = Munch({
        'maxtokens': 500,
        "pretrained": "cmarkea/distilcamembert-base",
        'context_size': 30,
        'unlabeled': True,
        'cuda': 0,
        'bsize':16,
        'device': None,
        'author_values': ['pad', 'alert', 'customer', 'operator'],
        'emotion_values': ['pad', 'no emotion', 'Surprise','Amusement','Satisfaction', 'Relief', 'Neutral','Fear','Sadness','Disappointment','Anger','Frustration'],
        'status_values' : ['Aborted', 'Solved', 'To be tested', 'Out of scope', 'No solution'],
        'satisfaction_values' : ['-3', '-2', '-1', '0', '1', '2', '3']
    })

with open(cmdargs.dataset, 'r', errors='ignore') as f:
            oui_json = {'train':list(), 'val':[], 'test':[], 'unlabeled': []}
            for i, line in enumerate(f):
                entry = json.loads(line)
                oui_json[entry['split']].append(entry)
print('file loaded')
label_names = args.author_values[2:]
num_labels = len(label_names)
label_dict = { args.author_values[i]:i for i in list(set([ idx for line in oui_json['train'] for idx in line['authors']  ])) }
print(label_dict)
label_distribution = [ args.author_values[a] for el in oui_json['train'] for a in el['authors'] ]
label_counts = Counter(label_distribution)
total_labels = float(len(label_distribution))
print('label counts', label_counts)
print('obtaining weights')
percentages = { l:label_counts[l] / total_labels for l in label_distribution }
print('percentages', percentages)
weights = [ 1-percentages[l] if l !='alert' else 0.0 for l in args.author_values[2:] ]
print('weights obtained')
print('ready to parse entries')


def _clean_pads(examples):
    '''remove paddings to enable a clean per message representation'''
    examples['texts'] = [ sent for sent in examples['texts'] if sent != '<pad>' ]
    examples['labels'] = [ emotion for emotion in examples['labels'] if emotion != 0]
    examples['authors'] = [ author for author in examples['authors'] if author != 0]
    return examples

def _parse_entry(entry, split):
    '''parse a dataset entry'''
    return _clean_pads(entry)['texts']

unlabeledset = {"text": [ message for entry in tqdm(oui_json['unlabeled'], desc="unlabeledset") for message in _parse_entry(entry, 'unlabeled')][:300] }

ds = Dataset.from_dict(unlabeledset)
ds = ds.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]], truncation=True)

tokenized_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=8,
    remove_columns=ds["train"].column_names,
)

block_size = 512

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=16)

## remove the last element (which has a different size)
for split in ['train', 'test']:
    indices = [i for i, el in enumerate(lm_dataset[split]) if len(el['input_ids']) == block_size ]
    lm_dataset[split] = lm_dataset[split].select(indices)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

model = AutoModelForMaskedLM.from_pretrained(args.pretrained)

print(model)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=20,
    weight_decay=0.01,
    save_steps=100
    # dataloader_drop_last=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

print(colored("second Masked Language Modelling pretraining for DAPT DONE", "green"))
