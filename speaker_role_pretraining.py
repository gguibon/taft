import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping # not used but may be useful if a huge dataset is used (as said in appendix)
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertModel
)
from tqdm import tqdm
from termcolor import colored
from munch import Munch

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from collections import Counter

from pytorch_lightning.loggers import TensorBoardLogger

print('imports done')

class CustomDataModule(LightningDataModule):
    def __init__(self, data_path: str = "data/dummy_dataset.json", batch_size: int = 32, num_workers=1, sep_token=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_labels = 0
        self.label_names = []
        self.task_name = "model_speaker_role_classification"
        self.sep_token = sep_token
        if self.sep_token is None:
            self.with_sep = False
            self.tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=True)
        else:
            self.with_sep = True
            self.tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=True, sep_token=self.sep_token)

        self.args = Munch({
            'maxtokens': 500,
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

    def prepare_data(self, seq_length=12, utt_length=20):
        with open(self.data_path, 'r', errors='ignore') as f:
            dataset_json = {'train':list(), 'val':[], 'test':[], 'unlabeled': []}
            for i, line in enumerate(f):
                entry = json.loads(line)
                dataset_json[entry['split']].append(entry) # ensures correct comparison
        print('file loaded')
        self.label_names = self.args.author_values[2:]
        self.num_labels = len(self.label_names)
        self.label_dict = { self.args.author_values[i]:i for i in list(set([ idx for line in dataset_json['train'] for idx in line['authors']  ])) }
        print(self.label_dict)
        label_distribution = [ self.args.author_values[a] for el in dataset_json['train'] for a in el['authors'] ]
        label_counts = Counter(label_distribution)
        total_labels = float(len(label_distribution))
        print('label counts', label_counts)
        print('obtaining weights')
        self.percentages = { l:label_counts[l] / total_labels for l in label_distribution }
        print('percentages', self.percentages)
        self.weights = [ 1-self.percentages[l] if l !='alert' else 0.0 for l in self.args.author_values[2:] ]
        print('weights obtained')
        print('ready to parse entries')


        def _clean_pads(examples):
            '''remove paddings to enable a clean per message representation'''
            examples['texts'] = [ sent for sent in examples['texts'] if sent != '<pad>' ]
            examples['labels'] = [ emotion for emotion in examples['labels'] if emotion != 0]
            examples['authors'] = [ author for author in examples['authors'] if author != 0]
            return examples

        def _parse_entry(entry, split, with_sep=False):
            '''parse a dataset entry and put it as the json bert format where each message is an item with a label (author).
            the conversation level is deconstruted
            split is not used as filtering is made before in dataset_json
            '''
            
            bert_jsons = [ self.tokenizer(e, padding="max_length", max_length=utt_length, truncation=True, add_special_tokens=True) for e in _clean_pads(entry)['texts'] ]
            for i, a in enumerate(entry['authors']): 
                bert_jsons[i].update({ 'labels': a-2, 'convid': int(entry['id'])  }) # -2 because we ignore pad and alert
                for k, v in bert_jsons[i].items(): bert_jsons[i][k] = torch.tensor(v)
            bert_jsons = [ obj for obj in bert_jsons if obj['labels'].item() >= 0]
            
            return bert_jsons

        self.trainset = [ message for entry in tqdm(dataset_json['train'], desc="trainset") for message in _parse_entry(entry, 'train')]
        self.valset = [ message for entry in tqdm(dataset_json['val'], desc="valset") for message in _parse_entry(entry, 'val') ]
        self.testset = [ message for entry in tqdm(dataset_json['test'], desc="testset") for message in _parse_entry(entry, 'test') ]
        self.unlabeledset = [ message for entry in tqdm(dataset_json['unlabeled'], desc="unlabeledset") for message in _parse_entry(entry, 'unlabeled')]

        self.unlabeled_trainset, self.unlabeled_valset, _, _ = train_test_split(self.unlabeledset, self.unlabeledset, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size)

    def unlabeled_dataloader(self):
        return DataLoader(self.unlabeledset, batch_size=self.batch_size)

    def labeled_dataloader(self):
        return DataLoader(self.trainset + self.valset + self.testset)

    def train_dataloader(self):
        return DataLoader(self.unlabeled_trainset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.unlabeled_valset, batch_size=self.batch_size)

class PLBert_direct(LightningModule):
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
        example_input_array=None,
        **kwargs,
    ):
        """
        Class for direct intermediate fine-tuning on speaker role (for "TAPT" or "TAFT no adapt" in the paper)
        """
        super().__init__()

        self.save_hyperparameters()

        self.example_input_array = example_input_array

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu =  torch.nn.ReLU()

        self.tanh = nn.Tanh()
        self.pool_fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

        self.num_labels = num_labels
        self.label_names = label_names
        self.label_weights = torch.tensor(label_weights, device=self.device)
        self.accuracy = Accuracy()
        self.f1 = F1(num_classes=num_labels, average='weighted')
        self.val_loss_list = []

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim)

        pooled_output = self.tanh(self.pool_fc(hidden_state[:, 0]))  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        
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
        loss = loss_fct(y_hat.view(-1, torch.tensor(self.num_labels, device=self.device)), label.view(-1) )
        return loss

    def on_fit_end(self) -> None:
        print(self.val_loss_list, type(self.val_loss_list))
        plt.clf()
        plt.plot(range(len(self.val_loss_list)), self.val_loss_list)
        plt.xlabel('epochs')
        plt.ylabel(self.optim.__class__.__name__)
        plt.title('lr: {}, N:{}, optim_alg:{}'.format(0, self.trainer.__annotations__, self.optim.__class__.__name__))
        plt.show()
        plt.savefig('%s/val_loss.png' % (self.trainer.logger.log_dir))

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
        self.val_loss_list.append(loss.detach().cpu().item())
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
        self.log("test_acc", self.accuracy, prog_bar=True)
        self.log("test_f1", self.f1, prog_bar=True)
        print('-'*5+'\n')
        print('labels', labels.size(), 'preds', preds.size(), 'len label_names', len(self.label_names), self.label_names)
        print('labels', labels[:1], 'preds', preds[:1] )
        print(classification_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), target_names=self.label_names, digits=4))
        cm = confusion_matrix(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
        df_cm = pd.DataFrame(cm, index=self.label_names, columns=self.label_names)
        plt.clf()
        sn.set(font_scale=1)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='coolwarm', linewidth=0.5, fmt="")
        plt.show()
        plt.savefig('%s/confusion_matrix.png' % (self.trainer.logger.log_dir))
        return loss

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
                {"params": self.classifier.parameters()}
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

class PLBert_adaptive(LightningModule):
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
        example_input_array=None,
        **kwargs,
    ):
        """ class for intermediate fine-tuning and pre-training of the adaptive layer (for "TAFT" and "TAFT noMTL" in the paper"""
        super().__init__()

        self.save_hyperparameters()

        self.example_input_array = example_input_array

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config)

        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu =  torch.nn.ReLU()

        self.tanh = nn.Tanh()
        self.pool_fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=12, batch_first=True)
        self.adaptive_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.num_labels = num_labels
        self.label_names = label_names
        self.label_weights = torch.tensor(label_weights, device=self.device)
        self.accuracy = Accuracy()
        self.f1 = F1(num_classes=num_labels, average='weighted')
        self.val_loss_list = []

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim)

        hidden_state = self.adaptive_layer(hidden_state)

        pooled_output = self.tanh(self.pool_fc(hidden_state[:, 0]))  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        
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
        loss = loss_fct(y_hat.view(-1, torch.tensor(self.num_labels, device=self.device)), label.view(-1)     )
        return loss

    def on_fit_end(self) -> None:
        print(self.val_loss_list, type(self.val_loss_list))
        plt.clf()
        plt.plot(range(len(self.val_loss_list)), self.val_loss_list)
        plt.xlabel('epochs')
        plt.ylabel(self.optim.__class__.__name__)
        plt.title('lr: {}, N:{}, optim_alg:{}'.format(0, self.trainer.__annotations__, self.optim.__class__.__name__))
        plt.show()
        plt.savefig('%s/val_loss.png' % (self.trainer.logger.log_dir))

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
        self.val_loss_list.append(loss.detach().cpu().item())
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
        self.log("test_acc", self.accuracy, prog_bar=True)
        self.log("test_f1", self.f1, prog_bar=True)
        print('-'*5+'\n')
        print('labels', labels.size(), 'preds', preds.size(), 'len label_names', len(self.label_names), self.label_names)
        print('labels', labels[:1], 'preds', preds[:1] )
        print(classification_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), target_names=self.label_names, digits=4))
        cm = confusion_matrix(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
        df_cm = pd.DataFrame(cm, index=self.label_names, columns=self.label_names)
        plt.clf()
        sn.set(font_scale=1)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='coolwarm', linewidth=0.5, fmt="")
        plt.show()
        plt.savefig('%s/confusion_matrix.png' % (self.trainer.logger.log_dir))
        return loss

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
                {
                    "params": self.adaptive_layer.parameters(),
                    "weight_decay_rate": 0.01,
                    "lr": 2e-5
                },
                {"params": self.pool_fc.parameters()},
                {"params": self.classifier.parameters()}
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
    parser = argparse.ArgumentParser(description="Speaker role prediction for either intermediate fine-tuning (TAPT) or domain-adaptation and adaptive layer pre-training (TAFT)")

    parser.add_argument("--type", type=str, default="adaptive",
                        help="Objective type"
                            "Options: [adaptive, direct]")
    parser.add_argument("--dataset", type=str, default="data/dummy_dataset.json",
                        help="Dataset path (default one is a dummy dataset with same format as the original (confidential)"
                            "Default: data/dummy_dataset.json")

    return parser.parse_args()

args = parse_args()

AVAIL_GPUS = min(1, torch.cuda.device_count())
dm = CustomDataModule(batch_size=128, num_workers=4*AVAIL_GPUS, data_path=args.dataset)
dm.prepare_data(utt_length=50)

seed_everything(42)

print(dm.num_labels, dm.label_names, dm.weights)


if args.type == "adaptive":
    tblogger = TensorBoardLogger("lightning_logs", name="%s_%s_%s" % (dm.task_name, "distilcamembert-base", "50_adaptive_1x12" ), log_graph=True)
    model = PLBert_adaptive(
        model_name_or_path="cmarkea/distilcamembert-base",
        num_labels=dm.num_labels,
        label_names=dm.label_names,
        task_name=dm.task_name,
        learning_rate=2e-5,
        label_weights=dm.weights,
    )
elif args.type == "direct":
    tblogger = TensorBoardLogger("lightning_logs", name="%s_%s_%s" % (dm.task_name, "distilcamembert-base", "50_directpooler" ), log_graph=True)
    model = PLBert_direct(
        model_name_or_path="cmarkea/distilcamembert-base",
        num_labels=dm.num_labels,
        label_names=dm.label_names,
        task_name=dm.task_name,
        learning_rate=2e-5,
        label_weights=dm.weights,
    )

checkpoint_callback = ModelCheckpoint( filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}', monitor='val_loss', mode="min" )

# model.summarize(mode='top')
AVAIL_GPUS = min(1, torch.cuda.device_count())
# you can set epochs to 10 due to higher values not improving anything (it sticks to ~93%). For TAPT, original paper uses 100 epochs, so we follow this.
trainer = Trainer(max_epochs=100, gpus=[0], logger=tblogger, enable_model_summary=True, callbacks=[checkpoint_callback], precision=32)
trainer.fit(model, datamodule=dm)

print(colored('restoring to best model:', 'yellow'), checkpoint_callback.kth_best_model_path)
model.load_from_checkpoint(checkpoint_callback.kth_best_model_path)
trainer.test(model, datamodule=dm)


# these lines are only useful to automate the model paths for the bash script
with open('temp.txt', 'w') as f:
    f.write(checkpoint_callback.kth_best_model_path)

print(colored("speaker role prediction DONE", "green"))
