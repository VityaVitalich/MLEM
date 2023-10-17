from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from configs.data_configs.physionet_inference import data_configs
from configs.model_configs.mTAN.physionet import model_configs
from src.data_load.dataloader import create_data_loaders
from src.models.base_models import GRUClassifier
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", help="path to folder with checkpoints")
    parser.add_argument("-d", help="device", default="cuda:0")
    args = parser.parse_args()

    conf = data_configs()
    model_conf = model_configs()
    train_loader, val_loader = create_data_loaders(conf)
    model = GRUClassifier(model_conf, conf).to(args.d)

    all_ckpt = Path(args.path)

    def key_fn(p: Path):
        return dict(it.split(": ") for it in p.stem.split(" - "))["total_loss"]

    p = min(all_ckpt.iterdir(), key=key_fn)
    ckpt = torch.load(p)["model"]
    model.load_state_dict(ckpt)

    @torch.no_grad()
    def calc_sim(loader):
        all_sim = []
        for batch, _ in tqdm(loader):
            batch = batch.to(args.d)
            emb = model(batch)
            sim = F.cosine_similarity(emb[:, None], emb[None], -1)
            idx = torch.triu_indices(*sim.shape, offset=1)
            all_sim.append(sim[idx[0], idx[1]])
        return torch.concat(all_sim).mean().item()

    train_sim = calc_sim(train_loader)
    val_sim = calc_sim(val_loader)
    print(f"Train similarity:\t{train_sim}")
    print(f"Valid similarity:\t{val_sim}")
