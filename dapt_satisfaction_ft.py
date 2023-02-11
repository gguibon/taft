import datasets, json, os, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1, CohenKappa
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertModel
)
# from transformers import get_scheduler
from datasets import load_dataset

from tqdm import tqdm
import numpy as np
from termcolor import colored
from collections import Counter
from munch import Munch

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from utils import tprint
import utils as utils
from sklearn.model_selection import train_test_split

from dataset_utils.imbalanced_sampler import ImbalancedDatasetSampler
from dataset_utils.custom_dataset import CustomDataset


class CustomDataModule(LightningDataModule):
    def __init__(self, data_path: str = "data/dummy_dataset.json", batch_size: int = 32, num_workers=1, task_satis="satis7", sep_token=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_labels = 0
        self.num_workers = num_workers
        self.label_names = []
        self.task_name = "dapt_satisfaction_clf"
        self.args = Munch({
            'maxtokens': 500,
            'context_size': 18,
            'unlabeled': True,
            'cuda': 0,
            'bsize':16,
            'device': None
        })
        self.task_satis = task_satis
        self.sep_token = sep_token
        if self.sep_token is None:
            self.with_sep = False
            self.tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=True)
        else:
            self.with_sep = True
            self.tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=True, sep_token=self.sep_token)

    def prepare_data(self, seq_length=12, utt_length=20):
        with open(self.data_path, 'r', errors='ignore') as f:
            ds_json = {'train':list(), 'val':[], 'test':[], 'unlabeled': []}
            for i, line in enumerate(f):
                entry = json.loads(line)
                ds_json[entry['split']].append(entry)

        self.label_names = ['-3', '-2', '-1', '0', '1', '2', '3']
        self.num_labels = len(self.label_names)
        self.label_dict = { self.label_names[i]:i for i in list(set([line['satisfaction'] for i, line in enumerate(ds_json['train'])])) }
        
        if self.task_satis == "satis3":
            self.label_polarity_names = ['negative', 'neutral', 'positive']
            self.num_polarity_labels = len(self.label_polarity_names)
            self.num_labels = self.num_polarity_labels

        label_distribution = [ self.label_names[el['satisfaction']] for el in ds_json['train'] if el['satisfaction'] != None]
        self.percentages = { l:Counter(label_distribution)[l] / float(len(label_distribution)) for l in label_distribution }
        self.weights = [ 1-self.percentages[l] if l != 'Aborted' else 0.0 for l in self.label_names ]

        if self.task_satis == "satis3": self.label_names = self.label_polarity_names

        def _clean_pads(examples):
            examples['texts'] = [ sent if sent != ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>'] else '[pad]' for sent in examples['texts'] ]
            return examples

        def _satisfaction2polarity(satisfaction):
            satisfaction = int(satisfaction)
            if satisfaction == 4: return self.label_polarity_names.index('neutral')
            elif satisfaction < 4: return self.label_polarity_names.index('negative')
            else: return self.label_polarity_names.index('positive')

        def _parse_entry(entry, split):
            '''parse a dataset entry and put it as the json bert format'''
            texts = self.tokenizer(' '.join([ e for e in _clean_pads(entry)['texts'] ]), padding="max_length", max_length=utt_length, truncation=True)
            if self.task_satis == "satis3": res = {'labels': _satisfaction2polarity(entry['satisfaction']) }
            else: res = {'labels': entry['satisfaction']}
            res.update(texts)
            for k, v in res.items(): res[k] = torch.tensor(v)
            return res

        self.trainset = [_parse_entry(entry, 'train') for entry in tqdm(ds_json['train'], desc="trainset")]
        self.valset = [_parse_entry(entry, 'val') for entry in tqdm(ds_json['val'], desc="valset")]
        self.testset = [_parse_entry(entry, 'test') for entry in tqdm(ds_json['test'], desc="testset")]

        self.trainset = [el for el in self.trainset if el['labels'] >= 0]
        self.valset = [el for el in self.valset if el['labels'] >= 0]
        self.testset = [el for el in self.testset if el['labels'] >= 0]
        
        num_samples = None
        self.train_batch_sampler = ImbalancedDatasetSampler(CustomDataset(self.trainset), num_samples=num_samples)
        self.val_batch_sampler = ImbalancedDatasetSampler(CustomDataset(self.valset), num_samples=num_samples)
        self.test_batch_sampler = ImbalancedDatasetSampler(CustomDataset(self.testset), num_samples=num_samples)

    def train_dataloader(self):
        return DataLoader(CustomDataset(self.trainset), pin_memory=True, sampler=self.train_batch_sampler, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(CustomDataset(self.valset), sampler=self.val_batch_sampler, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(CustomDataset(self.testset), sampler=self.test_batch_sampler, batch_size=self.batch_size)


class PLBert(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        label_names,
        label_weights,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits=None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.roberta = BertModel.from_pretrained(model_name_or_path, config=self.config)

        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu =  torch.nn.ReLU()

        self.tanh = nn.Tanh()
        self.pool_fc = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size)

        self.num_labels = num_labels
        self.label_names = label_names
        self.label_weights = torch.tensor(label_weights, device=self.device)
        self.accuracy = Accuracy()
        self.f1 = F1(num_classes=num_labels, average='weighted')
        self.kappa = CohenKappa(num_classes=num_labels)
        self.val_loss_list = []

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = self.tanh(self.pool_fc(hidden_state[:, 0]))  # (bs, dim)
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)  # (bs, dim)
        
        return logits
    
    def training_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        label = batch['labels']
        attention_mask = batch['attention_mask']
        
        y_hat = self(input_ids, attention_mask, label)
        
        percentages = { l:Counter(label.tolist())[l] / float(len(label)) for l in label.tolist() }
        weights = [ 1-percentages[l] if l in percentages else 0.0 for l in list(range(self.num_labels)) ]
        weights = torch.tensor(weights, device=self.device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(y_hat.view(-1, torch.tensor(self.num_labels, device=self.device)), label.view(-1))
        
        return loss

    def on_fit_end(self) -> None:
        print(self.val_loss_list, type(self.val_loss_list))
        plt.plot(range(len(self.val_loss_list)), self.val_loss_list)
        plt.xlabel('epochs')
        plt.ylabel(self.optim.__class__.__name__)
        plt.title('lr: {}, N:{}, optim_alg:{}'.format(0, self.trainer.__annotations__, self.optim.__class__.__name__))
        plt.show()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss_fct = torch.nn.CrossEntropyLoss()
        val_loss = loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.val_loss_list.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        self.accuracy(preds, labels)
        self.f1(preds, labels)
        self.log("val_acc", self.accuracy, prog_bar=True)
        self.log("val_f1", self.f1, prog_bar=True)
        print('-'*5+'\n')
        return loss

    def test_step(self, batch, batch_nb):
        input_ids = batch['input_ids']
        label = batch['labels']
        attention_mask = batch['attention_mask']
        logits = self(input_ids, attention_mask, label)

        loss_fct = torch.nn.CrossEntropyLoss()
        val_loss = loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("test_loss", loss, prog_bar=True)
        self.accuracy(preds, labels)
        self.f1(preds, labels)
        k_s = self.kappa(preds, labels)
        self.log("test_acc", self.accuracy, prog_bar=True)
        self.log("test_f1", self.f1, prog_bar=True)
        self.log("test_kappa", self.kappa, prog_bar=True)
        print('-'*5+'\n')
        print('labels', labels.size(), 'preds', preds.size(), 'len label_names', len(self.label_names))

        print(classification_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), target_names=self.label_names))
        cr = classification_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), 
        target_names=self.label_names, output_dict=True, digits=4)
        cr['kappa'] = k_s.cpu().numpy()
        utils.save_classificationreport(cr, '%s/classification_report.tsv' % (self.trainer.logger.log_dir) )
        print(colored("kappa", "green"), k_s)
        cm = confusion_matrix(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
        df_cm = pd.DataFrame(cm, index=self.label_names, columns=self.label_names)
        plt.clf()
        sn.set(font_scale=1)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='coolwarm', linewidth=0.5, fmt="")
        plt.show()
        plt.savefig('%s/confusion_matrix.png' % (self.trainer.logger.log_dir))
        return loss, cr["weighted avg"]

    def configure_optimizers(self):
        param_optimizer = list(self.roberta.named_parameters())
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
                {   "params": self.pool_fc.parameters()},
                {   "params": self.classifier.parameters()}
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                )
        self.optim = optimizer

        num_training_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        self.sched = scheduler
        return [optimizer], [scheduler]


def parse_args():
    parser = argparse.ArgumentParser(description="Second step for Domain-Adpative Pretraining (DAPT). Reused the MLM second pretraining for a finetuning on customer satisfaction prediction ")

    parser.add_argument("--dataset", type=str, default="data/dummy_dataset.json",
                        help="Dataset path (default one is a dummy dataset with same format as the original (confidential)"
                            "Default: data/dummy_dataset.json")
    parser.add_argument("--satis", type=str, default="satis3",
                        help="Satisfaction representation, either 3 polarities (satis3) or fine-grained 7 values (satis7)."
                            "Options: [satis3, satis7]")
    parser.add_argument("--modeltoload", type=str, required=True, help="path of the pretrained model using MLM.")

    return parser.parse_args()

args = parse_args()

seed_everything(42)

AVAIL_GPUS = min(1, torch.cuda.device_count())

dm = CustomDataModule(batch_size=16, num_workers=4*AVAIL_GPUS, task_satis=args.satis, data_path=args.dataset, sep_token="</s>")
dm.prepare_data(utt_length=512)
print('weights', dm.weights)
print('label names', dm.label_names)
print('num labels', dm.num_labels)
print('percentages', dm.percentages)
print('train label set', set([ l['labels'].item() for l in dm.trainset]) )
print('val label set', set([ l['labels'].item() for l in dm.valset]) )
print('test label set', set([ l['labels'].item() for l in dm.testset]) )

if args.satis == "satis3": tblogger = TensorBoardLogger("lightning_logs", name=dm.task_name.replace("satisfaction","satis3"), log_graph=True)
else: tblogger = TensorBoardLogger("lightning_logs", name=dm.task_name, log_graph=True)

model = PLBert(
    model_name_or_path="cmarkea/distilcamembert-base",
    num_labels=dm.num_labels,
    label_names=dm.label_names,
    task_name=dm.task_name,
    learning_rate=2e-5,
    label_weights=dm.weights,
    logger=tblogger,
    task_satis=args.satis
)

pretrained_model = torch.load(args.modeltoload)

### get dedicated weights from checkpoint into the model
model_keys = list(model.state_dict().keys())
pretrained_keys = list(pretrained_model.keys())
only_in_model = set(model_keys) - set(pretrained_keys)
only_in_pretrained = set(pretrained_keys) - set(model_keys)
for k in pretrained_keys:
    if k in ['lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', "lm_head.decoder.weight", "lm_head.decoder.bias"]:
        del pretrained_model[k]
for k in list(only_in_model): pretrained_model[k] = model.state_dict()[k]
print('='*20)
model.load_state_dict(pretrained_model)

### free some RAM
del pretrained_model

### Callbacks
checkpoint_callback = ModelCheckpoint( filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}', monitor='val_f1', mode="max" )

model.summarize(mode='top')
AVAIL_GPUS = min(1, torch.cuda.device_count())
trainer = Trainer(max_epochs=5, gpus=AVAIL_GPUS, logger=tblogger, enable_model_summary=True, callbacks=[checkpoint_callback], precision=32)
trainer.fit(model, datamodule=dm)

print(colored('restoring to best model:', 'yellow'), checkpoint_callback.kth_best_model_path)
model.load_from_checkpoint(checkpoint_callback.kth_best_model_path)
res = trainer.test(model, datamodule=dm)[0]



