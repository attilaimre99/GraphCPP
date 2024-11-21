# 5-fold Cross Validation

# Based on: https://gist.github.com/ashleve/ac511f08c0d29e74566900fd3efbb3ec

from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from graphcpp.dataset import CPPDataset
import lightning as L

class KFoldDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "dataset/",
            k: int = 1,  # fold number
            split_seed: int = 42,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False
        ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None

    @property
    def num_node_features() -> int:
        return 32

    @property
    def num_classes() -> int:
        return 2

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            dataset_full = CPPDataset(self.hparams.data_dir, fp_type='topological') # split train

            # choose fold to train on
            kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(dataset_full)]
            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            self.data_train, self.data_val = dataset_full[train_indexes], dataset_full[val_indexes]

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
    

from config import *
from graphcpp.lightning import GraphCPPModule

results = []
nums_folds = 5
split_seed = 42

for k in range(nums_folds):
    datamodule = KFoldDataModule(data_dir='dataset', k=k, num_splits=5, batch_size=128)
    datamodule.prepare_data()

    # Trainer options explanation in main.py
    trainer = L.Trainer(
        accelerator='cuda' if AVAIL_GPUS>0 else 'cpu',
        devices=AVAIL_GPUS if AVAIL_GPUS>0 else 'auto',
        max_epochs=50,
        enable_progress_bar=True,
        # precision="16-mixed",
        num_sanity_val_steps=0
    )

    # Create the model with the passed in arguments
    model = GraphCPPModule(**BEST_PARAMETERS)
    # Fit the trainer
    trainer.fit(model, datamodule=datamodule)
    # Load the best model checkpoint
    model = GraphCPPModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    # We validate the model and get the accuracy, MCC and AUC values
    scores = trainer.validate(model, datamodule=datamodule, verbose=True)

    # Append to the list of results
    results.append(scores[0])

print("Finished, results:")
print(results)