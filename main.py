import lightning as L
import torch
from config import *
L.seed_everything(RANDOM_SEED)
from graphcpp.lightning import GraphCPPModule, GraphCPPDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

def run_one_training(**model_kwargs):
    print(model_kwargs)

    # Create datamodule
    module = GraphCPPDataModule()

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        callbacks=[LearningRateMonitor("epoch"), ModelCheckpoint()],
        accelerator="cuda",
        devices=AVAIL_GPUS,
        max_epochs=38,
        enable_progress_bar=True,
        num_sanity_val_steps=0
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    model = GraphCPPModule.load_from_checkpoint(checkpoint_path="model/epoch=38-step=7020.ckpt")

    # Test model
    trainer.validate(model, datamodule=module, verbose=True)
    test_results = trainer.test(model, datamodule=module, verbose=True)
    return test_results[0]['test_mcc']
 
if __name__ == "__main__":
    model = run_one_training(**BEST_PARAMETERS)
    print(model)