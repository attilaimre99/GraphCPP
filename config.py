import torch

# Random seed
RANDOM_SEED = 42

# Cuda arguments
AVAIL_GPUS = min(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
BATCH_SIZE = 512

BEST_PARAMETERS = {
    'act': 'prelu',
    'conv_aggr': 'sum',
    'conv_dropout': 0.05,
    'has_bn': False,
    'has_l2norm': False,
    'layer_fingerprints': 1,
    'fingerprint_type': 'topological',
    'hidden_channels': 128,
    'layer_type': 'sageconv',
    'layers_pre_mp': 1,
    'mp_layers': 2,
    'layers_post_mp': 1,
    'learning_rate': 0.001,
    'pooling': 'mean',
    'stage_type': 'stack',
    'weight_decay': 'cosine'
}
