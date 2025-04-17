import torch.nn as nn
from deepsc.utils.freeze import freeze, unfreeze, count_trainable_params

def test_freeze_unfreeze():
    m = nn.Linear(10, 10)
    assert count_trainable_params(m) > 0
    freeze(m)
    assert count_trainable_params(m) == 0
    unfreeze(m)
    assert count_trainable_params(m) > 0
