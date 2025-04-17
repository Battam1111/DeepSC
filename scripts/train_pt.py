# -*- coding: utf-8 -*-
"""
不依赖 PyTorch‑Lightning 的朴素训练循环。
与 Lightning 共用 models/utils/data，无缝切换。
"""
import argparse, tqdm, torch, yaml
from deepsc.utils.seed import set_global_seed
from deepsc.data.europarl import make_dataloader
from deepsc.models.transformer import DeepSC
from deepsc.models import get_channel      # 自动触发 __init__.py
from deepsc.utils.power_norm import power_normalize
from deepsc.utils.mask import padding_mask, subsequent_mask

# ------------------ CLI ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True,
                    help='YAML 配置文件路径（同 Hydra，但这里手动解析）')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))

set_global_seed(cfg['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- DataLoader --------
train_loader = make_dataloader(cfg['data']['train_pkl'],
                               cfg['train']['batch_size'],
                               cfg['data']['pad_idx'])
# -------- Model / Channel --------
model = DeepSC(cfg['model']).to(device)
channel = get_channel(cfg['data']['channel'])()
criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg['data']['pad_idx'])
optim = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])

for epoch in range(cfg['train']['epochs']):
    pbar = tqdm.tqdm(train_loader)
    for batch in pbar:
        batch = batch.to(device)
        # --- forward ---
        n_var = 1 / (10 ** (cfg['data']['snr'] / 10) * 2) ** 0.5
        src_mask = padding_mask(batch, cfg['data']['pad_idx']).to(device)
        tgt_inp = batch[:, :-1]
        tgt_real = batch[:, 1:]

        enc = model.encoder(batch, src_mask)
        tx = power_normalize(model.channel_encoder(enc))
        rx = channel(tx, n_var)
        dec_out = model.decoder(tgt_inp, model.channel_decoder(rx),
                                subsequent_mask(tgt_inp.size(1)).to(device),
                                padding_mask(tgt_inp, cfg['data']['pad_idx']).to(device))
        pred = model.dense(dec_out)

        loss = criterion(pred.reshape(-1, pred.size(-1)), tgt_real.reshape(-1))
        # --- backward ---
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f'E{epoch}  loss={loss.item():.4f}')
