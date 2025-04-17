import torch
from deepsc.models.transformer import DeepSC
from deepsc.decoding.beam_search import beam_search
from deepsc.models.channel import AWGNChannel

def test_forward_and_beam():
    cfg = dict(vocab_size=100, max_len=32, n_layers=2,
               d_model=64, n_heads=4, d_ff=128,
               latent_dim=8, dropout=0.1, pad_idx=0)
    model = DeepSC(cfg)
    src = torch.randint(5, 99, (4, 10))
    n_var = 0.05
    logits = model(src, n_var, AWGNChannel())
    assert logits.shape[:2] == (4, 9)
    seqs = beam_search(model, src, n_var, AWGNChannel(),
                       1, 2, 0, beam_size=2, max_len=16)
    assert len(seqs) == 4
