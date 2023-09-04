import torch
from tqdm import tqdm
import pandas as pd
from .data_load.dataloader import create_data_loaders, create_test_loader
from .models.mTAND.model import MegaNetCE, MegaNet


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


def create_embeddings(data_inference_conf, model_conf):
    train_loader, valid_loader = create_data_loaders(data_inference_conf)
    test_loader = create_test_loader(data_inference_conf)

    net = MegaNetCE(model_conf=model_conf, data_conf=data_inference_conf).to(model_conf.device)

    ckpt = torch.load(data_inference_conf.ckpt_path)
    net.load_state_dict(ckpt["model"])

    valid_embeds, valid_indexes = get_embeds(valid_loader, net)
    save_embeds(valid_embeds, valid_indexes, data_inference_conf.valid_embed_path)
    print("valid embeds saved")

    test_embeds, test_indexes = get_embeds(test_loader, net)
    save_embeds(test_embeds, test_indexes, data_inference_conf.test_embed_path)
    print("test embeds saved")

    train_embeds, train_indexes = get_embeds(train_loader, net)
    save_embeds(train_embeds, train_indexes, data_inference_conf.train_embed_path)
    print("train embeds saved")
