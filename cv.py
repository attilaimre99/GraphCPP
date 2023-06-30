# 10-fold Cross Validation of best trained model

# From: https://gist.github.com/ashleve/ac511f08c0d29e74566900fd3efbb3ec

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
        return 30

    @property
    def num_classes() -> int:
        return 2

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            dataset_full = CPPDataset(self.hparams.data_dir) # split train

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
nums_folds = 10
split_seed = 42

for k in range(nums_folds):
    datamodule = KFoldDataModule(data_dir='dataset', k=k, batch_size=BATCH_SIZE)
    datamodule.prepare_data()
    datamodule.setup()

    # Trainer options explanation in main.py
    trainer = L.Trainer(
        accelerator='cuda' if AVAIL_GPUS>0 else 'cpu',
        devices=AVAIL_GPUS if AVAIL_GPUS>0 else 'auto',
        max_epochs=38,
        enable_progress_bar=False,
        precision="16-mixed",
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

# Results:
# {'val_accuracy': 0.8697916865348816, 'val_mcc': 0.7395877838134766, 'val_auc': 0.9374517798423767}
# {'val_accuracy': 0.875, 'val_mcc': 0.7540397047996521, 'val_auc': 0.9438223838806152}
# {'val_accuracy': 0.8817391395568848, 'val_mcc': 0.7695835828781128, 'val_auc': 0.9519333839416504}
# {'val_accuracy': 0.886956512928009, 'val_mcc': 0.7763028144836426, 'val_auc': 0.9515148401260376}
# {'val_accuracy': 0.8886956572532654, 'val_mcc': 0.7789894342422485, 'val_auc': 0.9549368023872375}
# {'val_accuracy': 0.8747826218605042, 'val_mcc': 0.7524040341377258, 'val_auc': 0.9432989358901978}
# {'val_accuracy': 0.9008695483207703, 'val_mcc': 0.801786482334137, 'val_auc': 0.954055905342102}
# {'val_accuracy': 0.8747826218605042, 'val_mcc': 0.7569807767868042, 'val_auc': 0.9468061923980713}
# {'val_accuracy': 0.8382608890533447, 'val_mcc': 0.6773791313171387, 'val_auc': 0.9257692694664001}
# {'val_accuracy': 0.8573912978172302, 'val_mcc': 0.7181056141853333, 'val_auc': 0.9478756189346313}