import lightning as L
from config import *
from datetime import datetime
L.seed_everything(RANDOM_SEED)
from graphcpp.lightning import GraphCPPModule, GraphCPPDataModule
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from mango import scheduler, Tuner as MangoTuner

# Decorator to run parallel
# Can be switched to @scheduler.serial for single runs
@scheduler.parallel(n_jobs=2)
def run_one_training(**model_kwargs):
    print(model_kwargs)

    # Create datamodule
    module = GraphCPPDataModule()

    # Create logger, default is tensorboard but here we use mlflow
    mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, run_name=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        logger=mlf_logger,
        callbacks=[
            EarlyStopping(monitor="val_mcc", mode="max", patience=10), # stop if the validation MCC value does not improve for 10 traning epochs
            ModelCheckpoint(monitor="val_mcc", mode="max")  # save the model with the highest validation MCC value
        ],
        accelerator="cuda",
        devices=AVAIL_GPUS,
        min_epochs=20,
        max_epochs=100,
        enable_progress_bar=False, # don't need
        precision="16-mixed", # for faster training, we don't loose much precision ref: https://lightning.ai/docs/pytorch/latest/common/precision_intermediate.html
        num_sanity_val_steps=0 # don't need
    )

    # Create the model with the passed in arguments
    model = GraphCPPModule(**model_kwargs)
    # Fit the trainer
    trainer.fit(model, datamodule=module)
    # Load the best model checkpoint
    model = GraphCPPModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Validate model and return val_mcc for arm-mango tuner
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
        "pooling": ["add", "mean", "max"],
        "conv_aggr": ["max", "mean", "softmax", "sum"],
        "act": ["relu", "prelu", "lrelu_01", "lrelu_025", "lrelu_05", "swish"],
        "has_bn": [True, False], # batch normalization
        "has_l2norm": [True, False], # l2 normalization
    }

    # Create arm-mango tuner with the objective of max val_mcc
    # num_iteration default: 20
    tuner = MangoTuner(HYPERPARAMETERS, objective=run_one_training, conf_dict={"num_iteration": 200})
    results = tuner.maximize()

    # Print the results
    print(results)
