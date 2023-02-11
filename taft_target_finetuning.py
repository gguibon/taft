from dataclasses import replace
import os, math, statistics, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datasets, json, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1, CohenKappa, Precision, Recall, MatthewsCorrCoef
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertModel
)
# from transformers import get_scheduler
from datasets import load_dataset
from datetime import datetime

from tqdm import tqdm
import numpy as np
from termcolor import colored
from collections import Counter
from munch import Munch

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, plot_roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from utils import tprint
import utils as utils
from sklearn.model_selection import train_test_split

from dataset_utils.imbalanced_sampler import ImbalancedDatasetSampler
from dataset_utils.custom_dataset import CustomDataset


class CustomDataModule(LightningDataModule):
    def __init__(self, data_path: str = "data/dummy_dataset.json", 
            batch_size: int = 32, 
            num_workers=1, 
            combine_mode="max_size_cycle",
            train_num_task_per_epoch = 1000,
            val_num_task: int = 600,
            test_num_task: int = 2000,
            way: int = 5,
            train_shot = 5,
            val_shot = 5,
            test_shot: int = 5,
            num_query: int = 15,
            drop_last = None,
            randomized_splits=False,
            sep_token=None,
            task_author=True,
            task_satis="satis3", # satis7
            task_status="status_noab" # status status_noab pc pc2
        ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.combine_mode = combine_mode
        self.num_labels = 0
        self.num_workers = num_workers
        self.label_names = []
        tasks = ['author' if task_author else '', task_satis, task_status]
        self.task_name = "speakerrole_%s_mtl_classification" % ( 'x'.join(tasks).strip('x') )
        self.sep_token = sep_token
        if self.sep_token is None:
            self.with_sep = False
            self.tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=True)
        else:
            self.with_sep = True
            self.tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=True, sep_token=self.sep_token)
            
        self.args = Munch({
            'maxtokens': 500,
            'context_size': 18,
            'unlabeled': True,
            'cuda': 0,
            'bsize':16,
            'device': None
        })
        self.train_num_task_per_epoch = train_num_task_per_epoch
        self.val_num_task = val_num_task
        self.test_num_task = test_num_task
        self.way = way
        self.train_shot = train_shot
        self.val_shot = val_shot
        self.test_shot = test_shot
        self.num_query = num_query
        self.drop_last = drop_last
        self.randomized_splits = randomized_splits
        self.task_status = task_status
        self.task_satis = task_satis
        self.task_author = task_author

    def prepare_data(self, seq_length=12, utt_length=20):
        with open(self.data_path, 'r', errors='ignore') as f:
            dataset_json = {'train':list(), 'val':[], 'test':[], 'unlabeled': []}
            for i, line in enumerate(f):
                entry = json.loads(line)
                dataset_json[entry['split']].append(entry) # ensures correct comparison

        self.label_status_names = ['Solved', 'To be tested', 'Out of scope', 'No solution']
        self.num_status_labels = len(self.label_status_names)
        self.label_status_dict = { self.label_status_names[i-1]:i for i in list(set([line['status'] for i, line in enumerate(dataset_json['train'])  if line['status'] != 'Aborted' ])) }
        

        self.label_satisfaction_names = ['-3', '-2', '-1', '0', '1', '2', '3']
        self.num_satisfaction_labels = len(self.label_satisfaction_names)
        self.label_satisfaction_dict = { self.label_satisfaction_names[i]:i for i in list(set([line['satisfaction'] for i, line in enumerate(dataset_json['train'])])) }

        if self.task_satis == "satis3":
            self.label_polarity_names = ['negative', 'neutral', 'positive']
            self.num_polarity_labels = len(self.label_polarity_names)

        self.label_author_names = ['pad', 'alert', 'customer', 'operator'][2:]
        self.num_author_labels = len(self.label_author_names)

        label_distribution = [ self.label_status_names[el['status'] -1] for el in dataset_json['train'] if el['status'] != None and el['status'] != 0]
        self.percentages = { l:Counter(label_distribution)[l] / float(len(label_distribution)) for l in label_distribution }
        self.weights = [ 1-self.percentages[l] if l != 'Aborted' else 0.0 for l in self.label_status_names ]

        if self.task_status == "pc":
            self.label_status_names = ["notproblematic", "problematic"]
            self.num_status_labels = len(self.label_status_names)

        def _clean_pads(examples):
            """
            original data was processed with padding, we remove it to handle padding by the hugging face tokenizer (in case it needs it)
            dummy data does not have this, hence this function does nothing on the dummy dataset
            """
            examples['texts'] = [ sent if sent != ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>'] else '[pad]' for sent in examples['texts'] ]
            return examples

        def _satisfaction2polarity(satisfaction):
            satisfaction = int(satisfaction)
            if satisfaction == 4: return self.label_polarity_names.index('neutral')
            elif satisfaction < 4: return self.label_polarity_names.index('negative')
            else: return self.label_polarity_names.index('positive')
        
        def _status2prob(status):
            status = int(status)
            if status <= 1: return self.label_status_names.index("notproblematic")
            elif status > 1: return self.label_status_names.index("problematic")

        def _parse_entry_status(entry, split, with_sep=False):
            '''parse a dataset entry and put it as the json bert format considering the status
            split is not used, as the entry is filtered by split before in dataset_json
            '''
            if with_sep:
                texts = self.tokenizer( self.tokenizer.sep_token.join(_clean_pads(entry)['texts']) , padding="max_length", max_length=utt_length, truncation=True)
            else: 
                texts = self.tokenizer(' '.join([ e for e in _clean_pads(entry)['texts'] ]), padding="max_length", max_length=utt_length, truncation=True)
            if self.task_status == "pc": res = {'labels': _status2prob(entry['status']-1)}
            else: res = {'labels': int(entry['status']) -1 }
            res.update(texts)
            for k, v in res.items(): res[k] = torch.tensor(v)
            return res

        def _parse_entry_satisfaction(entry, split, with_sep=False):
            '''parse a dataset entry and put it as the json bert format for the satisfaction'''
            if with_sep:
                texts = self.tokenizer( self.tokenizer.sep_token.join(_clean_pads(entry)['texts']) , padding="max_length", max_length=utt_length, truncation=True)
            else: 
                texts = self.tokenizer(' '.join([ e for e in _clean_pads(entry)['texts'] ]), padding="max_length", max_length=utt_length, truncation=True)
            if self.task_satis == "satis3": res = {'labels': _satisfaction2polarity(entry['satisfaction']) }
            else: res = {'labels': entry['satisfaction']}
            res.update(texts)
            for k, v in res.items(): res[k] = torch.tensor(v)
            return res

        def _parse_entry_author(entry, split):
            '''parse a dataset entry and put it as the json bert format where each message is an item with a label (author).
            the conversation level is deconstruted'''
            bert_jsons = [ self.tokenizer(e, padding="max_length", max_length=utt_length, truncation=True) for e in _clean_pads(entry)['texts'] ]
            for i, a in enumerate(entry['authors']): 
                bert_jsons[i].update({ 'labels': a-2, 'convid': int(entry['id'])  }) # -2 because we ignore pad and alert
                for k, v in bert_jsons[i].items(): bert_jsons[i][k] = torch.tensor(v)
            bert_jsons = [ obj for obj in bert_jsons if obj['labels'].item() >= 0]
            return bert_jsons

        if self.task_author:
            self.author_trainset = [ message for entry in tqdm(dataset_json['train'], desc="trainset") for message in _parse_entry_author(entry, 'train')]
            self.author_valset = [ message for entry in tqdm(dataset_json['val'], desc="valset") for message in _parse_entry_author(entry, 'val') ]
            self.author_testset = [ message for entry in tqdm(dataset_json['test'], desc="testset") for message in _parse_entry_author(entry, 'test') ]

        if self.task_status in ["status", "status_noab", "pc"]:
            self.status_trainset = [_parse_entry_status(entry, 'train', with_sep=self.with_sep) for entry in tqdm(dataset_json['train'], desc="trainset status") if entry['status'] > 0] # > 0 to avoid abordted conv
            self.status_valset = [_parse_entry_status(entry, 'val', with_sep=self.with_sep) for entry in tqdm(dataset_json['val'], desc="valset status") if entry['status'] > 0]
            self.status_testset = [_parse_entry_status(entry, 'test', with_sep=self.with_sep) for entry in tqdm(dataset_json['test'], desc="testset status") if entry['status'] > 0]
        
        if self.task_satis in ['satis3', 'satis7']:
            self.satisfaction_trainset = [_parse_entry_satisfaction(entry, 'train', with_sep=self.with_sep) for entry in tqdm(dataset_json['train'], desc="trainset satis") ]
            self.satisfaction_valset = [_parse_entry_satisfaction(entry, 'val', with_sep=self.with_sep) for entry in tqdm(dataset_json['val'], desc="valset satis") ]
            self.satisfaction_testset = [_parse_entry_satisfaction(entry, 'test', with_sep=self.with_sep) for entry in tqdm(dataset_json['test'], desc="testset satis") ]

        def _randomized_data_splits(datasets, seed=42):
            """useful for cross validation"""
            X = list(itertools.chain.from_iterable(datasets))
            y = [el['labels'] for el in X]
            X_train, X_test, y_train, _ = train_test_split(X , y, test_size=0.33, random_state=seed)
            X_train, X_val, _, _ = train_test_split( X_train, y_train, test_size=0.33, random_state=seed)
            return X_train, X_val, X_test

        if self.randomized_splits:
            # an option to use jointly with multiple fold; enables cross validation not stratified
            self.status_trainset, self.status_valset, self.status_testset = _randomized_data_splits(
                [self.status_trainset,self.status_valset,self.status_testset])
            self.satisfaction_trainset, self.satisfaction_valset, self.satisfaction_testset = _randomized_data_splits(
                [self.satisfaction_trainset,self.satisfaction_valset,self.satisfaction_testset])
            if self.task_author:
                self.author_trainset, self.author_valset, self.author_testset = _randomized_data_splits(
                    [self.author_trainset,self.author_valset,self.author_testset])


        num_samples = None
        self.train_batch_sampler_status = ImbalancedDatasetSampler(CustomDataset(self.status_trainset), num_samples=num_samples)
        self.val_batch_sampler_status = ImbalancedDatasetSampler(CustomDataset(self.status_valset), num_samples=num_samples)
        self.test_batch_sampler_status = ImbalancedDatasetSampler(CustomDataset(self.status_testset), num_samples=num_samples)

        self.train_batch_sampler_satisfaction = ImbalancedDatasetSampler(CustomDataset(self.satisfaction_trainset), num_samples=num_samples)
        self.val_batch_sampler_satisfaction = ImbalancedDatasetSampler(CustomDataset(self.satisfaction_valset), num_samples=num_samples)
        self.test_batch_sampler_satisfaction = ImbalancedDatasetSampler(CustomDataset(self.satisfaction_testset), num_samples=num_samples)

    def train_dataloader(self):
        loaders = {'status': DataLoader(CustomDataset(self.status_trainset), pin_memory=True, sampler=self.train_batch_sampler_status, batch_size=self.batch_size),
        'satis': DataLoader(CustomDataset(self.satisfaction_trainset), pin_memory=True, sampler=self.train_batch_sampler_satisfaction, batch_size=self.batch_size),
        }
        if self.task_author: 
            loaders['a'] = DataLoader(self.author_trainset, shuffle=True, batch_size=self.batch_size,  pin_memory=True)
        return CombinedLoader(loaders, 'max_size_cycle')

    def val_dataloader(self):
        loaders = {'status': DataLoader(CustomDataset(self.status_valset), num_workers=self.num_workers, sampler=self.val_batch_sampler_status, batch_size=self.batch_size),
        'satis': DataLoader(CustomDataset(self.satisfaction_valset), num_workers=self.num_workers, sampler=self.val_batch_sampler_satisfaction, batch_size=self.batch_size)
        }
        if self.task_author:
            loaders['a'] = DataLoader(self.author_valset, batch_size=self.batch_size, num_workers=self.num_workers)
        return CombinedLoader(loaders, 'min_size')

    def test_dataloader(self):
        loaders = {'status': DataLoader(CustomDataset(self.status_testset), num_workers=self.num_workers, sampler=self.test_batch_sampler_status, batch_size=self.batch_size),
        'satis': DataLoader(CustomDataset(self.satisfaction_testset), num_workers=self.num_workers, sampler=self.test_batch_sampler_satisfaction, batch_size=self.batch_size)
        }
        if self.task_author:
            loaders['a'] = DataLoader(self.author_testset, batch_size=self.batch_size, num_workers=self.num_workers)
        return CombinedLoader(loaders, 'min_size')


AVAIL_GPUS = min(1, torch.cuda.device_count())


class JointClf(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_status_labels: int,
        num_satisfaction_labels: int,
        num_author_labels: int,
        label_status_names,
        label_satisfaction_names,
        label_author_names,
        label_weights,
        task_name: str,
        learning_rate: float = 2e-5,
        task_author=True,
        testonly=False,
        adaptive=True,
        **kwargs,
    ):
        super().__init__()

        self.testonly = testonly
        self.save_hyperparameters()
        self.task_author = task_author
        self.adaptive = adaptive

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_satisfaction_labels)
        self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config)
        
        self.status_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_status_labels)
        self.satisfaction_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_satisfaction_labels)
        # self.author_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_author_labels)

        self.embedding_dim = self.bert.config.to_dict()['hidden_size']

        if self.adaptive:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=12, batch_first=True, dropout=0.1)
            self.adaptive_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.dropout = torch.nn.Dropout(0.5)
        self.relu =  torch.nn.ReLU()
        self.tanh = nn.Tanh()
        self.pool_fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

        self.num_status_labels = num_status_labels
        self.num_satisfaction_labels = num_satisfaction_labels
        self.num_author_labels = num_author_labels
        self.label_status_names = label_status_names
        self.label_satisfaction_names = label_satisfaction_names
        self.label_author_names = label_author_names
        self.label_weights = torch.tensor(label_weights, device=self.device)
        self.accuracy_status = Accuracy()
        self.accuracy_satisfaction = Accuracy()
        self.accuracy_author = Accuracy()
        self.kappa_status, self.kappa_satisfaction, self.kappa_author = CohenKappa(num_classes=num_status_labels), CohenKappa(num_classes=num_satisfaction_labels), CohenKappa(num_classes=num_author_labels)
        self.precision_status, self.precision_satisfaction, self.precision_author = Precision(num_classes=num_status_labels), Precision(num_classes=num_satisfaction_labels), Precision(num_classes=num_author_labels)
        self.recall_status, self.recall_satisfaction, self.recall_author = Recall(num_classes=num_status_labels), Recall(num_classes=num_satisfaction_labels), Recall(num_classes=num_author_labels)
        self.mcc_status, self.mcc_satisfaction, self.mcc_author = MatthewsCorrCoef(num_classes=num_status_labels), MatthewsCorrCoef(num_classes=num_satisfaction_labels), MatthewsCorrCoef(num_classes=num_author_labels)
        self.f1_status = F1(num_classes=num_status_labels, average='weighted')
        self.f1_satisfaction = F1(num_classes=num_satisfaction_labels, average='weighted')
        self.f1_author = F1(num_classes=num_author_labels, average='weighted')
        self.train_loss_list = []
        self.val_loss_list = []

    def forward(self, input_ids_status, attention_mask_status, input_ids_satisfaction, attention_mask_satisfaction, labels, input_ids_author, attention_mask_author):
        """episodic pseudo version"""
        logits_status = self._forward_status(input_ids_status, attention_mask_status)
        logits_satisfaction = self._forward_satisfaction(input_ids_satisfaction, attention_mask_satisfaction)
        if self.task_author:
            logits_author = self._forward_author(input_ids_author, attention_mask_author)
            return logits_status, logits_satisfaction, logits_author
        else:
            return logits_status, logits_satisfaction

    def _forward_status(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim) [20, 512, 768] outputs[1] = pooler

        if self.adaptive: hidden_state = self.adaptive_layer(hidden_state)
        pooled_output = self.tanh(self.pool_fc(hidden_state[:, 0]))  # (bs, dim)
        pooled_output = self.dropout(pooled_output)
        
        logits = self.status_classifier(pooled_output) # (bs, dim)
        return logits

    def _forward_satisfaction(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim) [20, 512, 768]
        
        if self.adaptive: hidden_state = self.adaptive_layer(hidden_state)
        pooled_output = self.tanh(self.pool_fc(hidden_state[:, 0]))  # (bs, dim)
        pooled_output = self.dropout(pooled_output)
        
        logits = self.satisfaction_classifier(pooled_output) # (bs, dim)
        return logits

    ## NOt USED but can be uncommented along with the classifier to enable multi task on 3 tasks: status, satisfaction, speaker role (here "author")
    # def _forward_author(self, input_ids, attention_mask):
    #     outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

    #     hidden_state = outputs[0]  # (bs, seq_len, dim)
    #     pooled_output = self.tanh(self.pool_fc(hidden_state[:, 0]))  # (bs, dim)
    #     pooled_output = self.dropout(pooled_output)  # (bs, dim)
        
    #     logits = self.author_classifier(pooled_output) # (bs, dim)
    #     return logits

    def training_step(self, batch, batch_idx):

        if self.task_author:
            y_hat_status, y_hat_satisfaction, y_hat_author = self(
            batch['status']['input_ids'], batch['status']['attention_mask'], 
            batch['satis']['input_ids'], batch['satis']['attention_mask'],
            (batch['status']['labels'], batch['satis']['labels'], batch['a']['labels']), 
            batch['a']['input_ids'], batch['a']['attention_mask']
            )
        else:
            y_hat_status, y_hat_satisfaction = self(
            batch['status']['input_ids'], batch['status']['attention_mask'], 
            batch['satis']['input_ids'], batch['satis']['attention_mask'],
            (batch['status']['labels'], batch['satis']['labels'], None), 
            None, None
            )
        status_loss = self._compute_celoss(y_hat_status, batch['status']['labels'], self.num_status_labels, weighted=True)
        satisfaction_loss = self._compute_celoss(y_hat_satisfaction, batch['satis']['labels'], self.num_satisfaction_labels, weighted=True)
        if self.task_author:
            author_loss = self._compute_celoss(y_hat_author, batch['a']['labels'], self.num_author_labels, weighted=True)
            loss = status_loss + satisfaction_loss + author_loss
        else: 
            tloss = [status_loss, satisfaction_loss]
            l_weights = [ 0.8, 0.2 ]
            loss = sum([tloss[i]*l_weights[i] for i in range(len(tloss))]) / sum(l_weights)

        return loss

    def _compute_celoss(self, y_hat, label, num_labels, weighted=False):
        """
            uses binary cross entropy if num_labels equals 2
        """
        if weighted:
            percentages = { l:Counter(label.tolist())[l] / float(len(label)) for l in label.tolist() }
            weights = [ 1-percentages[l] if l in percentages else 0.0 for l in list(range(num_labels)) ]
            weights = torch.tensor(weights, device=self.device)
            if num_labels > 2: 
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_fct(y_hat.view(-1, torch.tensor(num_labels, device=self.device)), label.view(-1))
            else:              
                loss_fct = torch.nn.BCEWithLogitsLoss(weight=weights)
                loss = loss_fct(y_hat.view(-1, torch.tensor(num_labels, device=self.device)).float(), F.one_hot(label.view(-1), num_classes=num_labels).float())
            
        else:
            if num_labels > 2: 
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(y_hat.view(-1, num_labels), label.view(-1))
            else:              
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(y_hat.view(-1, num_labels).float(), F.one_hot(label.view(-1), num_classes=num_labels).float())
        return loss

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.train_loss_list.append(loss.item())

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """episodic version"""
        if self.task_author:
            logits_status, logits_satisfaction, logits_author = self(
                batch['status']['input_ids'], batch['status']['attention_mask'],
                batch['satis']['input_ids'], batch['satis']['attention_mask'], 
                (batch['status']['labels'], batch['satis']['labels']), 
                batch['a']['input_ids'], batch['a']['attention_mask'])
        else:
            logits_status, logits_satisfaction = self(
                batch['status']['input_ids'], batch['status']['attention_mask'],
                batch['satis']['input_ids'], batch['satis']['attention_mask'], 
                (batch['status']['labels'], batch['satis']['labels']), 
                None, None)
        val_loss_status = self._compute_celoss(logits_status, batch['status']['labels'], self.num_status_labels, weighted=True)
        val_loss_satisfaction = self._compute_celoss(logits_satisfaction, batch['satis']['labels'], self.num_satisfaction_labels, weighted=True)
        if self.task_author:
            val_loss_author = self._compute_celoss(logits_author, batch['a']['labels'], self.num_author_labels)

        preds_status = torch.argmax(logits_status, axis=1)
        preds_satisfaction = torch.argmax(logits_satisfaction, axis=1)
        if self.task_author:
            preds_author = torch.argmax(logits_author, axis=1)

        res =  {"loss": val_loss_status+val_loss_satisfaction, "loss_status": val_loss_status, "loss_satisfaction": val_loss_satisfaction,  "preds_status": preds_status, "preds_satisfaction": preds_satisfaction, "labels_status": batch['status']['labels'], "labels_satisfaction": batch['satis']['labels']}
        if self.task_author:
            res.update({'labels_authors':batch['a']['labels'], "loss_author": val_loss_author, "preds_author": preds_author})

        return res

    def validation_epoch_end(self, outputs):
        preds_status = torch.cat([x["preds_status"] for x in outputs])
        preds_satisfaction = torch.cat([x["preds_satisfaction"] for x in outputs])
        if self.task_author: preds_author = torch.cat([x["preds_author"] for x in outputs])
        labels_status = torch.cat([x["labels_status"] for x in outputs])
        labels_satisfaction = torch.cat([x["labels_satisfaction"] for x in outputs])
        if self.task_author: labels_author = torch.cat([x["labels_author"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        loss_status = torch.stack([x["loss_status"] for x in outputs]).mean()
        loss_satisfaction = torch.stack([x["loss_satisfaction"] for x in outputs]).mean()
        if self.task_author: loss_author = torch.stack([x["loss_author"] for x in outputs]).mean()
        self.val_loss_list.append(loss.item())
        self.log("val_status_loss", loss_status)
        self.log("val_satisfaction_loss", loss_satisfaction)
        if self.task_author: self.log("val_author_loss", loss_author)
        self.log("val_loss", loss, prog_bar=True)
        if loss.item() < min(self.val_loss_list): self.log(colored("val_loss", "green"), loss, prog_bar=True)
        self.accuracy_status(preds_status, labels_status)
        self.accuracy_satisfaction(preds_satisfaction, labels_satisfaction)
        if self.task_author: self.accuracy_author(preds_author, labels_author)
        self.kappa_status(preds_status, labels_status)
        self.kappa_satisfaction(preds_satisfaction, labels_satisfaction)
        if self.task_author: self.kappa_author(preds_author, labels_author)
        self.f1_status(preds_status, labels_status)
        self.f1_satisfaction(preds_satisfaction, labels_satisfaction)
        if self.task_author: self.f1_author(preds_author, labels_author)
        self.log("val_acc_status", self.accuracy_status, prog_bar=True)
        self.log("val_f1_status", self.f1_status, prog_bar=True)
        self.log("val_kappa_status", self.kappa_status, prog_bar=True)
        self.log("val_acc_satisfaction", self.accuracy_satisfaction, prog_bar=True)
        self.log("val_f1_satisfaction", self.f1_satisfaction, prog_bar=True)
        self.log("val_kappa_satisfaction", self.kappa_satisfaction, prog_bar=True)
        if self.task_author: 
            self.log("val_acc_author", self.accuracy_author, prog_bar=True)
            self.log("val_f1_author", self.f1_author, prog_bar=True)
            self.log("val_kappa_author", self.kappa_author, prog_bar=True)
        print('-'*5+'\n')
        print()
        return loss

    def test_step(self, batch, batch_nb):
        if self.task_author:
            logits_status, logits_satisfaction, logits_author = self(
                batch['status']['input_ids'], batch['status']['attention_mask'],
                batch['satis']['input_ids'], batch['satis']['attention_mask'], 
                (batch['status']['labels'], batch['satis']['labels']), 
                batch['a']['input_ids'], batch['a']['attention_mask'])
        else:
            logits_status, logits_satisfaction = self(
            batch['status']['input_ids'], batch['status']['attention_mask'],
            batch['satis']['input_ids'], batch['satis']['attention_mask'], 
            (batch['status']['labels'], batch['satis']['labels']), 
            None, None)
        test_loss_status = self._compute_celoss(logits_status, batch['status']['labels'], self.num_status_labels, weighted=True)
        test_loss_satisfaction = self._compute_celoss(logits_satisfaction, batch['satis']['labels'], self.num_satisfaction_labels, weighted=True)
        if self.task_author: test_loss_author = self._compute_celoss(logits_author, batch['a']['labels'], self.num_author_labels)

        preds_status = torch.argmax(logits_status, axis=1)
        preds_satisfaction = torch.argmax(logits_satisfaction, axis=1)
        if self.task_author: preds_author = torch.argmax(logits_author, axis=1)

        res = {"loss": test_loss_status+test_loss_satisfaction, "preds_status": preds_status, "preds_satisfaction": preds_satisfaction,  "labels_status": batch['status']['labels'], "labels_satisfaction": batch['satis']['labels']}
        if self.task_author:
            res.update({ "preds_author": preds_author, "labels_author": batch['a']['labels'] })

        return res

    def test_epoch_end(self, outputs):
        preds_status = torch.cat([x["preds_status"] for x in outputs])
        preds_satisfaction = torch.cat([x["preds_satisfaction"] for x in outputs])
        labels_status = torch.cat([x["labels_status"] for x in outputs])
        labels_satisfaction = torch.cat([x["labels_satisfaction"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("test_loss", loss, prog_bar=True)
        
        self.accuracy_status(preds_status, labels_status)
        self.accuracy_satisfaction(preds_satisfaction, labels_satisfaction)
        
        self.f1_status(preds_status, labels_status)
        self.f1_satisfaction(preds_satisfaction, labels_satisfaction)

        self.precision_status(preds_status, labels_status)
        self.precision_satisfaction(preds_satisfaction, labels_satisfaction)

        self.recall_status(preds_status, labels_status)
        self.recall_satisfaction(preds_satisfaction, labels_satisfaction)

        self.mcc_status(preds_status, labels_status)
        self.mcc_satisfaction(preds_satisfaction, labels_satisfaction)

        k_p = self.kappa_status(preds_status, labels_status)
        k_s = self.kappa_satisfaction(preds_satisfaction, labels_satisfaction)
        
        self.log("test_acc_status", self.accuracy_status, prog_bar=True)
        self.log("test_f1_status", self.f1_status, prog_bar=True)
        self.log("test_kappa_status", self.kappa_status, prog_bar=True)
        self.log("test_precision_status", self.precision_status, prog_bar=True)
        self.log("test_recall_status", self.recall_status, prog_bar=True)
        self.log("test_mcc_status", self.mcc_status, prog_bar=True)
        
        self.log("test_acc_satisfaction", self.accuracy_satisfaction, prog_bar=True)
        self.log("test_f1_satisfaction", self.f1_satisfaction, prog_bar=True)
        self.log("test_kappa_satisfaction", self.kappa_satisfaction, prog_bar=True)
        self.log("test_precision_satisfaction", self.precision_satisfaction, prog_bar=True)
        self.log("test_recall_satisfaction", self.recall_satisfaction, prog_bar=True)
        self.log("test_mcc_satisfaction", self.mcc_satisfaction, prog_bar=True)


        if self.task_author: 
            preds_author = torch.cat([x["preds_author"] for x in outputs])
            labels_author = torch.cat([x["labels_author"] for x in outputs])
            self.accuracy_author(preds_author, labels_author)
            self.f1_author(preds_author, labels_author)
            self.kappa_author(preds_author, labels_author)
            self.log("test_acc_author", self.accuracy_author, prog_bar=True)
            self.log("test_f1_author", self.f1_author, prog_bar=True)
            self.log("test_kappa_author", self.kappa_author, prog_bar=True)
        print('-'*5+'\n')
        print('labels_status', labels_status.size(), 'preds_status', preds_status.size(), 'len label_status_names', len(self.label_status_names))
        print('labels_satisfaction', labels_satisfaction.size(), 'preds_satisfaction', preds_satisfaction.size(), 'len label_satisfaction_names', len(self.label_satisfaction_names))

        labels_preds = list(set(preds_status.detach().tolist()))
        print(colored("status", "cyan"), "status", list(set(labels_status.detach().tolist())), "preds", labels_preds, self.label_status_names)
        labels_preds = None
        print(classification_report(labels_status.detach().cpu().numpy(), preds_status.detach().cpu().numpy(), target_names=self.label_status_names, labels=labels_preds, digits=4))
        cr = classification_report(labels_status.detach().cpu().numpy(), preds_status.detach().cpu().numpy(), target_names=self.label_status_names, output_dict=True, labels=labels_preds, digits=4)
        cr['kappa'] = k_p.cpu().numpy()
        utils.save_classificationreport(cr, '%s/classification_report_status.tsv' % (self.trainer.logger.log_dir) )
        print(colored("kappa status", "green"), k_p)
        cm = confusion_matrix(labels_status.detach().cpu().numpy(), preds_status.detach().cpu().numpy())
        df_cm = pd.DataFrame(cm, index=self.label_status_names, columns=self.label_status_names)
        plt.clf()
        sn.set(font_scale=1)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='coolwarm', linewidth=0.5, fmt="")
        plt.show()
        plt.savefig('%s/confusion_matrix_status.png' % (self.trainer.logger.log_dir))
        cr_status = cr['weighted avg']

        print('labels_satisfaction', labels_satisfaction.size(), 'preds_satisfaction', preds_satisfaction.size(), 'len label_satisfaction_names', len(self.label_satisfaction_names))
        print(classification_report(labels_satisfaction.detach().cpu().numpy(), preds_satisfaction.detach().cpu().numpy(), target_names=self.label_satisfaction_names, digits=4))
        cr = classification_report(labels_satisfaction.detach().cpu().numpy(), preds_satisfaction.detach().cpu().numpy(), target_names=self.label_satisfaction_names, output_dict=True, digits=4)
        cr['kappa'] = k_s.cpu().numpy()
        utils.save_classificationreport(cr, '%s/classification_report_satisfaction.tsv' % (self.trainer.logger.log_dir) )
        print(colored("kappa satisfaction", "green"), k_s)
        cm = confusion_matrix(labels_satisfaction.detach().cpu().numpy(), preds_satisfaction.detach().cpu().numpy())
        df_cm = pd.DataFrame(cm, index=self.label_satisfaction_names, columns=self.label_satisfaction_names)
        plt.clf()
        sn.set(font_scale=1)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='coolwarm', linewidth=0.5, fmt="")
        plt.show()
        plt.savefig('%s/confusion_matrix_satisfaction.png' % (self.trainer.logger.log_dir))
        cr_satisfaction = cr['weighted avg']

        if self.task_author: 
            labels_preds = list(set(preds_author.detach().tolist()))
            print('labels_author', labels_author.size(), 'preds_author', preds_author.size(), 'len label_author_names', len(self.label_author_names))
            print(classification_report(labels_author.detach().cpu().numpy(), preds_author.detach().cpu().numpy(), target_names=self.label_author_names, labels=labels_preds, digits=4))
            cr = classification_report(labels_author.detach().cpu().numpy(), preds_author.detach().cpu().numpy(), target_names=self.label_author_names, output_dict=True, labels=labels_preds, digits=4)
            utils.save_classificationreport(cr, '%s/classification_report_author.tsv' % (self.trainer.logger.log_dir) )
            print(colored("kappa author", "green"), self.kappa_author)
            cm = confusion_matrix(labels_author.detach().cpu().numpy(), preds_author.detach().cpu().numpy())
            df_cm = pd.DataFrame(cm, index=self.label_author_names, columns=self.label_author_names)
            plt.clf()
            sn.set(font_scale=1)
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='coolwarm', linewidth=0.5, fmt="")
            plt.show()
            plt.savefig('%s/confusion_matrix_author.png' % (self.trainer.logger.log_dir))
            cr_author = cr['weighted avg']

        res = {"status":cr_status, "satisfaction":cr_satisfaction}
        if self.task_author: res['author'] = cr_author

        if not self.testonly:
            print('loss lists', self.train_loss_list, self.val_loss_list)
            plt.clf()
            plt.plot(range(len(self.val_loss_list)), self.val_loss_list)
            plt.plot(range(len(self.train_loss_list)), self.train_loss_list)
            plt.xlabel('epochs')
            plt.ylabel(self.optim.__class__.__name__)
            plt.title('lr: {}, N:{}, optim_alg:{}'.format(0, self.trainer.__annotations__, self.optim.__class__.__name__))
            plt.show()
            plt.savefig('%s/trval_loss.png' % (self.trainer.logger.log_dir))
        
        return loss, res

    def configure_optimizers(self):
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                {"params": self.pool_fc.parameters()},
                {"params": self.status_classifier.parameters()},
                {"params": self.satisfaction_classifier.parameters()}
                ]
        if self.adaptive:
            optimizer_grouped_parameters.append({
                        "params": self.adaptive_layer.parameters(),
                        "weight_decay_rate": 0.00001,
                        "lr": 1e-3
                    })
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate
                )
        self.optim = optimizer

        num_training_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        self.sched = scheduler

        return [optimizer], [scheduler]



def parse_args():
    parser = argparse.ArgumentParser(description="Speaker role prediction for either intermediate fine-tuning (TAPT) or domain-adaptation and adaptive layer pre-training (TAFT)")

    parser.add_argument("--dataset", type=str, default="data/dummy_dataset.json",
                        help="Dataset path (default one is a dummy dataset with same format as the original (confidential)"
                            "Default: data/dummy_dataset.json")
    parser.add_argument("--type", type=str, default="adaptive",
                        help="Objective type"
                            "This leads to the TAFT noAdapt (direct), TAFT (adaptive) or TAPT (tapt) from Figure 1 in the paper"
                            "model path in modeltoload option should possess the correct structure"
                            "Options: [adaptive, direct, tapt]")
    parser.add_argument("--satis", type=str, default="satis3",
                        help="Satisfaction representation, either 3 polarities (satis3) or fine-grained 7 values (satis7)."
                            "Options: [satis3, satis7]")
    parser.add_argument("--status", type=str, default="pc",
                        help="Status representation, either as Problem Statuses (without aborted conversations) or binary problematic conversation (PC)"
                            "Options: [status, pc]")
    parser.add_argument("--modeltoload", type=str, required=True, help="path of the pretrained model using speaker role identification."
                            "it should have the same structure to enable correct weights transfer (adaptive layer if any, etc.)")
    parser.add_argument("--lr", type=float, default=5e-5, 
                        help="Learning rate.")
    parser.add_argument("--testonly", action="store_true", default=False, help="do not train and only test the model to obtain its standard dev over 5 runs"
                            "requires the model_path to be the fully trained one (the 2 steps from Figure 1 in the paper)")

    return parser.parse_args()

args = parse_args()

seed_everything(42)

task_author = False
folds = 1
status_f1_avg, satisfaction_f1_avg, author_f1_avg = [], [], []
for fold in list(range(folds)):

    dm = CustomDataModule(batch_size=32, num_workers=4*AVAIL_GPUS, sep_token="</s>", task_author=task_author, task_satis=args.satis, task_status=args.status, data_path=args.dataset)
    dm.prepare_data(utt_length=512)

    print('train label sets', Counter([ l['labels'].item() for l in dm.status_trainset]), Counter([ l['labels'].item() for l in dm.satisfaction_trainset]) )

    print("%s_%sdistilcamembert_50_adaptive" % ( dm.task_name,  "sep_" if dm.with_sep else ""))
    if folds > 1: tblogger = TensorBoardLogger("lightning_logs", name="cv_%s_%sdistilcamembert_50" % ( dm.task_name,  "sep_" if dm.with_sep else ""), log_graph=True, version=f'fold_{fold + 1}')
    else: tblogger = TensorBoardLogger("lightning_logs", name="%s_%sdistilcamembert_50_adaptive" % ( dm.task_name,  "sep_" if dm.with_sep else ""), log_graph=True)

    num_satis = dm.num_satisfaction_labels
    satis_names = dm.label_satisfaction_names
    if args.satis == "satis3":
        num_satis = dm.num_polarity_labels
        satis_names = dm.label_polarity_names
    
    model = JointClf(
            model_name_or_path="cmarkea/distilcamembert-base",
            num_status_labels=dm.num_status_labels,
            num_satisfaction_labels=num_satis,
            num_author_labels=dm.num_author_labels,
            label_status_names=dm.label_status_names,
            label_satisfaction_names=satis_names,
            label_author_names=dm.label_author_names,
            task_name=dm.task_name.replace('_author_',''),
            learning_rate=args.lr,
            label_weights=dm.weights,
            logger=tblogger,
            task_author=task_author,
            adaptive=args.type=="adaptive"
        )

    if args.testonly:
        checkpoint_callback = ModelCheckpoint( filename='{epoch}-{val_f1_status:.2f}-{val_loss:.2f}', monitor='val_f1_status', mode="max" )
        model.summarize(mode='top')
        AVAIL_GPUS = min(1, torch.cuda.device_count())
        # we create the trainer because the model uses some inherent params, however we do not train it
        trainer = Trainer(max_epochs=5, gpus=AVAIL_GPUS, logger=tblogger, enable_model_summary=True, precision=32, callbacks=[checkpoint_callback] )
        model = model.load_from_checkpoint(args.modeltoload)
        model.testonly = True
        
    else:

        pretrained_model = torch.load(args.modeltoload)
        print("="*10)
        print(pretrained_model['state_dict'].keys())
        print(pretrained_model.keys())
        print("="*10)

        ### get dedicated weights from checkpoint into the model
        model_keys = list(model.state_dict().keys())
        pretrained_keys = list(pretrained_model['state_dict'].keys())
        only_in_model = set(model_keys) - set(pretrained_keys)
        only_in_pretrained = set(pretrained_keys) - set(model_keys)
        
        for k in pretrained_keys:
            if k in ['classifier.weight', 'classifier.bias']:
                del pretrained_model['state_dict'][k]
            if k in ['status_classifier.weight', 'status_classifier.bias', 'satisfaction_classifier.weight', 'satisfaction_classifier.bias', 'author_classifier.weight', 'author_classifier.bias']: 
                pretrained_model['state_dict'][k] = model.state_dict()[k]
            
        for k in list(only_in_model): pretrained_model['state_dict'][k] = model.state_dict()[k]
        model.load_state_dict(pretrained_model['state_dict'])
        
        
        if args.type in ["adaptive","direct"]:
            ## Freeze the BERT layers for TAFT, but not for TAPT
            for param in model.bert.parameters(): param.requires_grad = False

        ### free some RAM
        del pretrained_model

        ### Callbacks
        checkpoint_callback = ModelCheckpoint( filename='{epoch}-{val_f1_status:.2f}-{val_loss:.2f}', monitor='val_f1_status', mode="max" )

        model.summarize(mode='top')
        AVAIL_GPUS = min(1, torch.cuda.device_count())
        trainer = Trainer(max_epochs=2, gpus=AVAIL_GPUS, logger=tblogger, enable_model_summary=True, precision=16, callbacks=[checkpoint_callback] )
        trainer.fit(model, datamodule=dm)

        print(colored('restoring to best model:', 'yellow'), checkpoint_callback.kth_best_model_path)
        model.load_from_checkpoint(checkpoint_callback.kth_best_model_path)
    
    res = trainer.test(model, datamodule=dm)[0]
    status_f1_avg.append(res['test_f1_status']); satisfaction_f1_avg.append(res['test_f1_satisfaction'])
    if task_author: author_f1_avg.append(res['test_f1_author'])

if task_author:
    print( colored("status f1", "cyan"), sum(status_f1_avg)/len(status_f1_avg),
    colored("satisfaction f1", "cyan"), sum(satisfaction_f1_avg)/len(satisfaction_f1_avg),
    colored("author f1", "cyan"), sum(author_f1_avg)/len(author_f1_avg) )
else: 
    if len(status_f1_avg) > 1: print( colored("status f1", "cyan"), sum(status_f1_avg)/len(status_f1_avg), "std", statistics.stdev(status_f1_avg),
        colored("satisfaction f1", "cyan"), sum(satisfaction_f1_avg)/len(satisfaction_f1_avg), "std", statistics.stdev(satisfaction_f1_avg) )

if args.testonly:
    ## creates 5 test runs using a fully trained TAFT to obtain the standard deviation (as given in the paper, Table 3)
    def calculate_std(nb_runs=5):
        st_f1_avg, satis_f1_avg = [], []
        results = []
        for i in range(nb_runs):
            res = trainer.test(model, datamodule=dm)[0]
            results.append(res)
        print("="*5, "status std", "="*5)
        print("acc %s precision %s recall %s f1 %s kappa %s mcc %s" % (
            statistics.stdev([r["test_acc_status"] for r in results]),
            statistics.stdev([r["test_precision_status"] for r in results]),
            statistics.stdev([r["test_recall_status"] for r in results]),
            statistics.stdev([r["test_f1_status"] for r in results]),
            statistics.stdev([r["test_kappa_status"] for r in results]),
            statistics.stdev([r["test_mcc_status"] for r in results])
        ))
        print("="*5, "satisfaction std", "="*5)
        print("acc %s precision %s recall %s f1 %s kappa %s mcc %s" % (
            statistics.stdev([r["test_acc_satisfaction"] for r in results]),
            statistics.stdev([r["test_precision_satisfaction"] for r in results]),
            statistics.stdev([r["test_recall_satisfaction"] for r in results]),
            statistics.stdev([r["test_f1_satisfaction"] for r in results]),
            statistics.stdev([r["test_kappa_satisfaction"] for r in results]),
            statistics.stdev([r["test_mcc_satisfaction"] for r in results])
        ))

    calculate_std(nb_runs=5)




