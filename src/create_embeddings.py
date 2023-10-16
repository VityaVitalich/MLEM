import torch
from tqdm import tqdm
import pandas as pd
from .data_load.dataloader import create_data_loaders, create_test_loader
from .models import base_models


@torch.no_grad()
def get_embeds(loader, net):
    all_embeddings = []
    ids = []
    for batch in tqdm(loader):
        inp, gt = batch
        inp, gt = inp.to("cuda:3"), gt.to("cuda:3")
        out = net(inp)

        batch_size = len(inp)
        embeddings = out["z"].view(batch_size, -1)
        all_embeddings.append(embeddings)
        ids.append(gt)

    all_embeds = torch.cat(all_embeddings)
    all_indices = torch.cat(ids)

    return all_embeds, all_indices


def save_embeds(embeds, indices, save_path):
    df = pd.DataFrame(data=embeds.cpu().numpy(), index=indices.cpu().numpy())
    df.to_csv(save_path)


def predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for inp, gt in tqdm(loader):
            gts.append(gt.to(device))
            inp = inp.to(device)
            pred = model(inp)
            preds.append(pred)

    return preds, gts


def create_embeddings(data_inference_conf, model_conf):
    data_inference_conf.valid_size = 0
    data_inference_conf.train.split_strategy = {"split_strategy": "NoSplit"}
    train_supervised_loader, valid_loader = create_data_loaders(
        data_inference_conf, supervised=True
    )
    test_loader = create_test_loader(data_inference_conf)

    model = getattr(base_models, model_conf.model_name)
    net = model(model_conf=model_conf, data_conf=data_inference_conf)

    ckpt = torch.load(data_inference_conf.ckpt_path)
    net.load_state_dict(ckpt["model"])
    net = net.to(model_conf.device)

    train_embeddings, train_gts = predict(
        net, train_supervised_loader, model_conf.device
    )
    test_embeddings, test_gts = predict(net, test_loader, model_conf.device)

    return train_embeddings, train_gts, test_embeddings, test_gts
