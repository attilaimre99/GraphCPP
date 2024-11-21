import lightning as L
from config import RANDOM_SEED, BEST_PARAMETERS, AVAIL_GPUS
from graphcpp.lightning import GraphCPPModule, GraphCPPDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from datetime import datetime
import argparse

def run_one_training(experiment_name, dataset, **model_kwargs):
    mlf_logger = MLFlowLogger(experiment_name="Default", tracking_uri="file:./mlruns", run_name=f"{experiment_name}")

    # Create datamodule for the current fold
    module = GraphCPPDataModule(folder=dataset, fp_type=model_kwargs['fingerprint_type'])
    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        # callbacks=[EarlyStopping(monitor="val_mcc", mode="max", patience=20), ModelCheckpoint(monitor="val_mcc", mode="max")],
        callbacks=[ModelCheckpoint(monitor="val_mcc", mode="max")],
        accelerator='cuda' if AVAIL_GPUS > 0 else 'cpu',
        devices=AVAIL_GPUS if AVAIL_GPUS > 0 else 'auto',
        max_epochs=100,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        logger=mlf_logger
    )

    model = GraphCPPModule(**model_kwargs)
    trainer.fit(model, train_dataloaders=module.train_dataloader(), val_dataloaders=module.val_dataloader())
    model = GraphCPPModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Validate and test the model
    trainer.validate(model, dataloaders=module, verbose=True)
    test_results = trainer.test(model, dataloaders=module, verbose=True)
    return test_results[0]['test_mcc']
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine learning models")
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    start_time = datetime.now()
    parser.add_argument('--experiment_name', type=str, help='Experiment name', default=f"{start_time}")
    # dataset path folder
    parser.add_argument('--dataset', type=str, help='Dataset folder', default='dataset')
    args = parser.parse_args()

    L.seed_everything(args.seed)
    print(f"Seed: {args.seed}")

    print(f"Start time: {start_time}")

    result = run_one_training(args.experiment_name, args.dataset, **BEST_PARAMETERS)
    print(f"Test MCC: {result}")