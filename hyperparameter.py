import lightning as L
from config import *
L.seed_everything(RANDOM_SEED)
from graphcpp.lightning import GraphCPPModule, GraphCPPDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from mango import scheduler, Tuner as MangoTuner

# Can be switched to @scheduler.serial for single runs
@scheduler.parallel(n_jobs=2)
def run_one_training(**model_kwargs):
    print(model_kwargs)

    # Create datamodule
    module = GraphCPPDataModule()

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        callbacks=[LearningRateMonitor("epoch"),EarlyStopping(monitor="val_mcc", mode="max", patience=10), ModelCheckpoint(monitor="val_mcc", mode="max")],
        accelerator="cuda",
        devices=AVAIL_GPUS,
        min_epochs=20,
        max_epochs=100,
        enable_progress_bar=False,
        # deterministic=True,
        precision="16-mixed",
        num_sanity_val_steps=0
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    model = GraphCPPModule(**model_kwargs)
    trainer.fit(model, datamodule=module)
    model = GraphCPPModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Validate model
    validation_results = trainer.validate(model, datamodule=module, verbose=False)
    return validation_results[0]['val_mcc']

if __name__ == "__main__":
    HYPERPARAMETERS = {
        "learning_rate": [0.001, 0.01, 0.1],
        "layers_pre_mp": range(1, 4),
        "mp_layers": range(1, 8),
        "layers_post_mp": range(0, 4),
        "hidden_channels": [256,512],
        "layer_type": ['sageconv'],
        "stage_type": ["skipconcat", "skipsum", "stack"],
        "conv_dropout": [0.0, 0.3, 0.6],
        "pooling": ["add", "mean", "max"], # left in the variables that hadn't been tested with the old architecture
        "conv_aggr": ["max", "mean", "softmax", "sum"], # left in the variables that hadn't been tested with the old architecture
        "act": ["relu", "prelu", "lrelu_01", "lrelu_025", "lrelu_05", "swish"], # left in the variables that hadn't been tested with the old architecture
        "has_bn": [True, False],
        "has_l2norm": [True, False],
    }

    tuner = MangoTuner(HYPERPARAMETERS, objective=run_one_training, conf_dict={"num_iteration": 200})
    results = tuner.maximize()

    print(results)
