import lightning as L
import torch
from config import *
L.seed_everything(RANDOM_SEED)
from graphcpp.lightning import GraphCPPModule, GraphCPPDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def run_one_training(**model_kwargs):
    print(model_kwargs)

    # Create datamodule
    module = GraphCPPDataModule(fp_type=model_kwargs['fingerprint_type'])

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        # callbacks=[ModelCheckpoint(monitor="val_mcc", mode="max")],
        # callbacks=[EarlyStopping(monitor="val_mcc", mode="max", patience=10), ModelCheckpoint(monitor="val_mcc", mode="max")],
        accelerator='cuda' if AVAIL_GPUS>0 else 'cpu',
        devices=AVAIL_GPUS if AVAIL_GPUS>0 else 'auto',
        max_epochs=50,
        enable_progress_bar=True,
        num_sanity_val_steps=0
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    # if os.path.isfile("model/epoch=38-step=7020.ckpt"):
    # model = GraphCPPModule.load_from_checkpoint(checkpoint_path="model/checkpoints/epoch=50-step=1173.ckpt")
    # else:
    model = GraphCPPModule(**model_kwargs)
    trainer.fit(model, datamodule=module)
    model = GraphCPPModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test model
    trainer.validate(model, datamodule=module, verbose=True)
    test_results = trainer.test(model, datamodule=module, verbose=True)
    return test_results[0]['test_mcc']
 
if __name__ == "__main__":
    model = run_one_training(**BEST_PARAMETERS)
    print(model)