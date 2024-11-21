import torch
from torch_geometric.loader import DataLoader
from graphcpp.dataset import CPPDataset
from graphcpp.model import GCN
import lightning as L
from pytorch_lightning import LightningDataModule
from config import *
import torchmetrics
from sklearn.model_selection import KFold
from typing import Optional
from torch.utils.data import Subset
from datetime import datetime

class GraphCPPKFoldDataModule(LightningDataModule):
    def __init__(
            self,
            folder: str = "dataset",
            fp_type: str = 'type',
            k: int = 1,  # fold number
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 5,
            batch_size: int = BATCH_SIZE,
            num_workers: int = 0,
            pin_memory: bool = False
        ):
        super().__init__()
        
        self.save_hyperparameters(logger=False)

        assert 1 <= k <= num_splits, "incorrect fold number"
        
        self.transforms = None

        self.data_train: Optional[Subset] = None
        self.data_val: Optional[Subset] = None
        self.data_test = CPPDataset(root=folder, _split='test', fp_type=fp_type).shuffle()

    @property
    def num_node_features(self) -> int:
        return 128  # Update this to your number of features

    @property
    def num_classes(self) -> int:
        return 1  # Update this to your number of classes

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            dataset_train = CPPDataset(root=self.hparams.folder, _split='train', fp_type=self.hparams.fp_type).shuffle()
            dataset_val = CPPDataset(root=self.hparams.folder, _split='val', fp_type=self.hparams.fp_type).shuffle()
            dataset_full = dataset_train + dataset_val

            kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(dataset_full)]
            train_indexes, val_indexes = all_splits[self.hparams.k - 1]

            self.data_train = Subset(dataset_full, train_indexes)
            self.data_val = Subset(dataset_full, val_indexes)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)


        # return DataLoader(self.mlcpp2_independent, batch_size=BATCH_SIZE)

    # def __init__(self, folder='dataset', **kwargs):
    #     super().__init__()
    #     self.train_split = CPPDataset(root=folder, _split='train', fp_type=kwargs['fp_type']).shuffle()
    #     self.val_split = CPPDataset(root=folder, _split='val', fp_type=kwargs['fp_type']).shuffle()
    #     self.test_split = CPPDataset(root=folder, _split='test', fp_type=kwargs['fp_type']).shuffle()

    # def train_dataloader(self):
    #     return DataLoader(self.train_split, batch_size=BATCH_SIZE, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_split, batch_size=BATCH_SIZE)

    # def test_dataloader(self):
    #     return DataLoader(self.test_split, batch_size=BATCH_SIZE)

class GraphCPPDataModule(L.LightningDataModule):
    def __init__(self, folder='dataset', fp_type='topological', **kwargs):
        super().__init__()
        self.train_split = CPPDataset(root=folder, _split='train', fp_type=fp_type).shuffle()
        self.val_split = CPPDataset(root=folder, _split='val', fp_type=fp_type).shuffle()
        self.test_split = CPPDataset(root=folder, _split='test', fp_type=fp_type).shuffle()
        # self.mlcpp2_independent = CPPDataset(root=folder, _split='mlcpp2_independent', fp_type=fp_type).shuffle()

        # print the test split
        for i, data in enumerate(self.mlcpp2_independent):
            print(f"{i}_{data['y'].item()}_{data['name']}")

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mlcpp2_independent, batch_size=BATCH_SIZE)

class GraphCPPModule(L.LightningModule):
    def __init__(self, learning_rate=0.001, weight_decay=None, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = GCN(**model_kwargs)

        self.loss_module = torch.nn.BCEWithLogitsLoss()
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_mcc = torchmetrics.MatthewsCorrCoef(task='binary')
        self.val_auc = torchmetrics.AUROC(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_mcc = torchmetrics.MatthewsCorrCoef(task='binary')
        self.test_auc = torchmetrics.AUROC(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.test_recall = torchmetrics.Recall(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

    def forward(self, data):
        return self.model(data) 

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def configure_optimizers(self):
        if self.weight_decay == None:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.weight_decay == "cosine":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        preds, labels = self(batch)
        preds = preds.squeeze(1)
        labels = labels.float()
        loss = self.loss_module(preds, labels)
        self.log("train_loss", loss, batch_size=len(batch))
        print(f"Train loss: {loss}")
        # Accuracy
        self.train_accuracy(preds, labels)
        self.log('train_accuracy', self.train_accuracy, batch_size=len(batch))
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, labels = self(batch)
        preds = preds.squeeze(1)
        labels = labels.float()
        self.val_accuracy(preds, labels)
        self.val_mcc(preds, labels)
        self.val_auc(preds, labels)
        # validation loss
        loss = self.loss_module(preds, labels)
        print(f"Validation loss: {loss}")
        print(f"Validation MCC: {self.val_mcc.compute()}")
        self.log("val_loss", loss, batch_size=len(batch))
        self.log('val_accuracy', self.val_accuracy, batch_size=len(batch))
        self.log('val_mcc', self.val_mcc, batch_size=len(batch))
        self.log('val_auc', self.val_auc, batch_size=len(batch))

    def test_step(self, batch, batch_idx):
        preds, labels = self(batch)
        # log as a file artifact the predictions and labels into mlflow

        preds = preds.squeeze(1)
        labels = labels.float()

        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(f'.\comparison-predictions\{current_time}.csv', 'w') as f:
            preds_cpu = preds.cpu().detach().numpy()
            f.write('probability\n')
            for i in range(len(preds_cpu)):
                f.write(f"{preds_cpu[i].item()}\n")

        self.test_accuracy(preds, labels)
        self.test_mcc(preds, labels)
        self.test_auc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        self.log('test_accuracy', self.test_accuracy, batch_size=len(batch))
        self.log('test_mcc', self.test_mcc, batch_size=len(batch))
        self.log('test_auc', self.test_auc, batch_size=len(batch))
        self.log('test_precision', self.test_precision, batch_size=len(batch))
        self.log('test_recall', self.test_recall, batch_size=len(batch))
        self.log('test_f1', self.test_f1, batch_size=len(batch))
