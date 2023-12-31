import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from graphcpp.dataset import CPPDataset
# from graphcpp.model_for_tsne import GCN
from graphcpp.model import GCN
import lightning as L
from config import *
import torchmetrics

# Load datasets
class GraphCPPDataModule(L.LightningDataModule):
    def __init__(self, folder='dataset', **kwargs):
        super().__init__()
        self.train_split = CPPDataset(root=folder, _split='train', fp_type=kwargs['fp_type']).shuffle()
        self.val_split = CPPDataset(root=folder, _split='val', fp_type=kwargs['fp_type']).shuffle()
        self.test_split = CPPDataset(root=folder, _split='test', fp_type=kwargs['fp_type']).shuffle()
        self.mlcpp2_independent = CPPDataset(root=folder, _split='mlcpp2_independent', fp_type=kwargs['fp_type']).shuffle()

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=BATCH_SIZE)
        # return DataLoader(self.mlcpp2_independent, batch_size=BATCH_SIZE)

class GraphCPPModule(L.LightningModule):
    def __init__(self, learning_rate=0.001, weight_decay=0.0005, **model_kwargs):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def training_step(self, batch, batch_idx):
        preds, labels = self(batch)
        preds = preds.squeeze(1)
        labels = labels.float()
        loss = self.loss_module(preds, labels)
        self.log("train_loss", loss, batch_size=len(batch))
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
        self.log('val_accuracy', self.val_accuracy, batch_size=len(batch))
        self.log('val_mcc', self.val_mcc, batch_size=len(batch))
        self.log('val_auc', self.val_auc, batch_size=len(batch))

    def test_step(self, batch, batch_idx):
        preds, labels = self(batch)
        preds = preds.squeeze(1)
        labels = labels.float()
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
