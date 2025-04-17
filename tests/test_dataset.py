import pytest, torch
from deepsc.data.europarl import EuroParlDataset

@pytest.fixture()
def tiny_dataset(tmp_path):
    path = tmp_path / 'toy.pkl'
    import pickle
    pickle.dump([[1,2,3],[4,5]], open(path, 'wb'))
    return path

def test_len_and_iter(tiny_dataset):
    ds = EuroParlDataset(tiny_dataset)
    assert len(ds) == 2
    rows = list(iter(ds))
    assert all(isinstance(r, torch.Tensor) for r in rows)
