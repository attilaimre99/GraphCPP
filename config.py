import torch

# Random seed
RANDOM_SEED = 42

# Cuda arguments
AVAIL_GPUS = min(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
BATCH_SIZE = 256

BEST_PARAMETERS = {
    'act': 'prelu',
    'conv_aggr': 'softmax',
    'conv_dropout': 0.0,
    'has_bn': True,
    'has_l2norm': False,
    'layer_fingerprints': 2,
    'fingerprint_type': 'topological',
    'hidden_channels': 256,
    'layer_type': 'sageconv',
    'layers_pre_mp': 1,
    'mp_layers': 2,
    'layers_post_mp': 3,
    'learning_rate': 0.001,
    'pooling': 'mean',
    'stage_type': 'stack',
    'weight_decay': 0.0005
}