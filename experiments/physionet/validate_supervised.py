import torch
from tqdm import tqdm
import pandas as pd
import sys

sys.path.append("../../")
from src.data_load.dataloader import create_data_loaders, create_test_loader
from src.models.mTAND.model import MegaNetClassifier
from src.models.base_models import GRUClassifier as model_type
from configs.data_configs.physionet import data_configs
from configs.model_configs.mTAN.physionet import model_configs
from sklearn.metrics import roc_auc_score
from src.trainers.randomness import seed_everything


@torch.no_grad()
def get_preds(loader, net):
    net.eval()
    preds, gt = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            out = net(batch[0].to(net.model_conf.device))
            preds.append(out.cpu()[:, 1])
            gt.append(batch[1][1])

    return torch.cat(preds), torch.cat(gt)


def validate_model(data_inference_conf, model_conf):
    seed_everything(
        conf.client_list_shuffle_seed,
        avoid_benchmark_noise=True,
        only_deterministic_algorithms=False,
    )

    train_loader, valid_loader = create_data_loaders(data_inference_conf)
    test_loader = create_test_loader(data_inference_conf)

    net = model_type(model_conf=model_conf, data_conf=data_inference_conf).to(
        model_conf.device
    )

    ckpt = torch.load(data_inference_conf.ckpt_path)
    net.load_state_dict(ckpt["model"])

    preds, y_test = get_preds(test_loader, net)
    score = roc_auc_score(y_test, preds)
    print(score)


if __name__ == "__main__":
    conf = data_configs()
    model_conf = model_configs()
    validate_model(data_inference_conf=conf, model_conf=model_conf)