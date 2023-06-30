import torch

# Random seed
RANDOM_SEED = 42

# MLFLOW settings
EXPERIMENT_NAME = "Default"
# If mlflow RUN_ID and CHECKPOINT_ID is set than the model doesn"t start training but instead loads the experiment.
MLFLOW_RUN_ID = None
MLFLOW_CHECKPOINT_NAME = None

# Cuda arguments
AVAIL_GPUS = min(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
BATCH_SIZE = 32

BEST_PARAMETERS = {
    'act': 'relu',
    'conv_aggr': 'max',
    'conv_dropout': 0.6,
    'has_bn': True,
    'has_l2norm': False,
    'hidden_channels': 256,
    'layer_type': 'sageconv',
    'layers_post_mp': 3,
    'layers_pre_mp': 3,
    'learning_rate': 0.001,
    'mp_layers': 1,
    'pooling': 'mean',
    'stage_type': 'skipconcat',
    'weight_decay': 0.0005
}
# own test set mcc: 0.813 at 38 epochs with modelchekcpoint val_mcc max
# MLCPP2 test set mcc: 0.664 at 38 epochs with modelchekcpoint val_mcc max
