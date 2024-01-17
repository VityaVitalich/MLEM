import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_almost_equal
from .sampling_strategies import PairSelector, get_sampling_strategy
from typing import Literal, Union


class BaseContrastiveLoss(nn.Module):
    def __init__(
        self,
        project_dim: Union[int, None] = None,
        projector: Literal["Identity", "Linear", "MLP"] = "Identity",
    ):
        super().__init__()

        if projector == "Identity":
            self.net = nn.Identity()
        elif projector == "Linear":
            assert project_dim is not None
            self.net = nn.LazyLinear(project_dim)
        elif projector == "MLP":
            assert project_dim is not None
            self.net = nn.Sequential(
                nn.LazyLinear(project_dim),
                nn.ReLU(inplace=True),
                nn.LazyLinear(project_dim),
            )
        else:
            raise ValueError(
                f"Unkown projector: {projector}. "
                "Valid values are: {identiy, linear, MLP}"
            )

    def project(self, x):
        return self.net(x)


class ContrastiveLoss(BaseContrastiveLoss):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, pair_selector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        embeddings = self.project(embeddings)

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target
        )
        positive_loss = F.pairwise_distance(
            embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]
        ).pow(2)

        negative_loss = F.relu(
            self.margin
            - F.pairwise_distance(
                embeddings[negative_pairs[:, 0]],
                embeddings[negative_pairs[:, 1]],
            )
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.sum()  # / (len(positive_pairs) + len(negative_pairs))


class InfoNCELoss(BaseContrastiveLoss):
    """
    InfoNCE Loss https://arxiv.org/abs/1807.03748
    """

    def __init__(
        self,
        temperature: float,
        pair_selector: PairSelector,
        angular_margin: float = 0.0,  # = 0.5 ArcFace default
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.pair_selector = pair_selector
        self.angular_margin = angular_margin

    def forward(self, embeddings, target):
        embeddings = self.project(embeddings)

        positive_pairs, _ = self.pair_selector.get_pairs(embeddings, target)
        dev = positive_pairs.device
        all_idx = torch.arange(len(positive_pairs), dtype=torch.long, device=dev)
        mask_same = torch.zeros(len(positive_pairs), len(embeddings), device=dev)
        mask_same[all_idx, positive_pairs[:, 0]] -= torch.inf

        sim = (
            F.cosine_similarity(
                embeddings[positive_pairs[:, 0], None],
                embeddings[None],
                dim=-1,
            )
            + mask_same
        )
        if self.angular_margin > 0.0:
            with torch.no_grad():
                target_sim = sim[all_idx, positive_pairs[:, 1]].clamp(0, 1)
                target_sim.arccos_()
                target_sim += self.angular_margin
                target_sim.cos_()
                sim[all_idx, positive_pairs[:, 1]] = target_sim

        sim /= self.temperature
        lsm = -F.log_softmax(sim, dim=-1)
        loss = torch.take_along_dim(
            lsm,
            positive_pairs[:, [1]],
            dim=1,
        ).sum()
        return loss


class RINCELoss(BaseContrastiveLoss):
    """
    Robust Contrastive Learning against Noisy Views
    """

    def __init__(
        self,
        temperature: float,
        pair_selector: PairSelector,
        q: float,
        lam: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.pair_selector = pair_selector
        self.q = q
        self.lam = lam

    def forward(self, embeddings, target):
        embeddings = self.project(embeddings)

        positive_pairs, _ = self.pair_selector.get_pairs(embeddings, target)
        dev = positive_pairs.device
        all_idx = torch.arange(len(positive_pairs), dtype=torch.long, device=dev)
        mask_same = torch.zeros(len(positive_pairs), len(embeddings), device=dev)
        mask_same[all_idx, positive_pairs[:, 0]] -= torch.inf

        sim = (
            F.cosine_similarity(
                embeddings[positive_pairs[:, 0], None],
                embeddings[None],
                dim=-1,
            )
            + mask_same
        )
        sim = torch.exp(sim / self.temperature)
        pos = torch.take_along_dim(
            sim,
            positive_pairs[:, [1]],
            dim=1,
        )
        all_ = sim.sum(1)
        loss = -(pos**self.q) / self.q + (self.lam * all_) ** self.q / self.q

        return loss.sum()


class DecoupledInfoNCELoss(BaseContrastiveLoss):
    """
    Inspired by https://arxiv.org/abs/2110.06848
    -log((sum exp of positive pairs) / (sum exp of negative pairs))
    """

    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, embeddings, target):
        embeddings = self.project(embeddings)

        pos = target[:, None] == target
        neg_mask = torch.where(pos, -torch.inf, 0)
        pos_mask = torch.where(pos, 0, -torch.inf) - torch.diag(
            torch.full(target.shape, torch.inf, device=pos.device)
        )

        sim = (
            F.cosine_similarity(
                embeddings[:, None],
                embeddings[None],
                dim=-1,
            )
            / self.temperature
        )
        loss = torch.logsumexp(sim + neg_mask, dim=1) - torch.logsumexp(
            sim + pos_mask, dim=1
        )

        return loss.sum()


class DecoupledPairwiseInfoNCELoss(BaseContrastiveLoss):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, temperature, pair_selector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        embeddings = self.project(embeddings)

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target
        )
        pos = target[positive_pairs[:, 0], None] == target
        neg_mask = torch.where(pos, -torch.inf, 0)
        sim = (
            F.cosine_similarity(
                embeddings[positive_pairs[:, 0], None],
                embeddings[None],
                dim=-1,
            )
            / self.temperature
        )
        pos_sim = sim[positive_pairs[:, 0], positive_pairs[:, 1]][:, None]
        neg_sim = sim[positive_pairs[:, 0]] + neg_mask - pos_sim
        loss = torch.logsumexp(neg_sim, dim=1)

        return loss.sum()


class BinomialDevianceLoss(nn.Module):
    """
    Binomial Deviance loss

    "Deep Metric Learning for Person Re-Identification", ICPR2014
    https://arxiv.org/abs/1407.4979
    """

    def __init__(self, pair_selector, alpha=1, beta=1, C=1):
        super(BinomialDevianceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target
        )

        pos_pair_similarity = F.cosine_similarity(
            embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]], dim=1
        )
        neg_pair_similarity = F.cosine_similarity(
            embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]], dim=1
        )

        pos_loss = torch.mean(
            torch.log(1 + torch.exp(-self.alpha * (pos_pair_similarity - self.beta)))
        )
        neg_loss = torch.mean(
            torch.log(
                1 + torch.exp(self.alpha * self.C * (neg_pair_similarity - self.beta))
            )
        )

        res_loss = (pos_loss + neg_loss) * (len(target))

        return res_loss, len(positive_pairs) + len(negative_pairs)


class TripletLoss(nn.Module):
    """
    Triplets loss

    "Deep metric learning using triplet network", SIMBAD 2015
    https://arxiv.org/abs/1412.6622
    """

    def __init__(self, margin, triplet_selector):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = F.pairwise_distance(
            embeddings[triplets[:, 0]], embeddings[triplets[:, 1]]
        )
        an_distances = F.pairwise_distance(
            embeddings[triplets[:, 0]], embeddings[triplets[:, 2]]
        )
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.sum(), len(triplets)


class HistogramLoss(torch.nn.Module):
    """
    HistogramLoss

    "Learning deep embeddings with histogram loss", NIPS 2016
    https://arxiv.org/abs/1611.00822
    code based on https://github.com/valerystrizh/pytorch-histogram-loss
    """

    def __init__(self, num_steps=100, device="cuda"):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.device = device
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        self.t = self.t.to(self.device)

    def forward(self, embeddings, classes):
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (
                (s_repeat_floor - (self.t - self.step) > -self.eps)
                & (s_repeat_floor - (self.t - self.step) < self.eps)
                & inds
            )

            assert (
                indsa.nonzero(as_tuple=False).size()[0] == size
            ), "Another number of bins should be used"
            zeros = torch.zeros((1, indsa.size()[1])).bool()
            zeros = zeros.to(self.device)
            indsb = torch.cat((indsa, zeros))[1:, :]
            s_repeat_[~(indsb | indsa)] = 0
            # indsa corresponds to the first condition of the second equation of the paper
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
            # indsb corresponds to the second condition of the second equation of the paper
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step

            return s_repeat_.sum(1) / size

        # L2 normalization
        classes_size = classes.size()[0]
        classes_eq = (
            classes.repeat(classes_size, 1)
            == classes.view(-1, 1).repeat(1, classes_size)
        ).data
        dists = outer_cosine_similarity(embeddings)

        assert (
            (dists > 1 + self.eps).sum().item() + (dists < -1 - self.eps).sum().item()
        ) == 0, ("L2 normalization " "should be used ")
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).bool()
        s_inds = s_inds.to(self.device)
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s = s.clamp(-1 + 1e-6, 1 - 1e-6)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (
            torch.floor((s_repeat.data + 1.0 - 1e-6) / self.step) * self.step - 1.0
        ).float()

        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(
            histogram_pos.sum().item(),
            1,
            decimal=1,
            err_msg="Not good positive histogram",
            verbose=True,
        )
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(
            histogram_neg.sum().item(),
            1,
            decimal=1,
            err_msg="Not good negative histogram",
            verbose=True,
        )
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(
            1, histogram_pos.size()[0]
        )
        histogram_pos_inds = torch.tril(
            torch.ones(histogram_pos_repeat.size()), -1
        ).bool()
        histogram_pos_inds = histogram_pos_inds.to(self.device)
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss, pos_size + neg_size


class MarginLoss(torch.nn.Module):

    """
    Margin loss

    "Sampling Matters in Deep Embedding Learning", ICCV 2017
    https://arxiv.org/abs/1706.07567

    """

    def __init__(self, pair_selector, margin=1, beta=1.2):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.beta = beta
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target
        )

        d_ap = F.pairwise_distance(
            embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]
        )
        d_an = F.pairwise_distance(
            embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]]
        )

        pos_loss = torch.clamp(d_ap - self.beta + self.margin, min=0.0)
        neg_loss = torch.clamp(self.beta - d_an + self.margin, min=0.0)

        loss = torch.cat([pos_loss, neg_loss], dim=0)

        return loss.sum(), len(positive_pairs) + len(negative_pairs)


class ComplexLoss(torch.nn.Module):
    def __init__(self, ml_loss, aug_loss, ml_loss_weight=1.0):
        super(ComplexLoss, self).__init__()
        self.aug_loss = aug_loss
        self.ml_loss = ml_loss
        self.ml_loss_weight = ml_loss_weight

    def forward(self, model_ouputs, target):
        aug_output, ml_output = model_ouputs
        aug_target = target[:, 0]
        ml_target = target[:, 1]
        aug = self.aug_loss(aug_output, aug_target) * (1 - self.ml_loss_weight)
        ml = self.ml_loss(ml_output, ml_target) * self.ml_loss_weight
        return aug + ml


def get_loss(model_conf):
    sampling_strategy = get_sampling_strategy(model_conf)

    if model_conf.loss.loss_fn == "CrossEntropy":
        loss_fn = torch.nn.CrossEntropyLoss()

    elif model_conf.loss.loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss()

    elif model_conf.loss.loss_fn == "ContrastiveLoss":
        kwargs = {
            "margin": model_conf.loss.margin,
            "pair_selector": sampling_strategy,
            "projector": model_conf.loss.projector,
            "project_dim": model_conf.loss.project_dim,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = ContrastiveLoss(**kwargs)

    elif model_conf.loss.loss_fn == "InfoNCELoss":
        kwargs = {
            "temperature": model_conf.loss.temperature,
            "pair_selector": sampling_strategy,
            "angular_margin": model_conf.loss.angular_margin,
            "projector": model_conf.loss.projector,
            "project_dim": model_conf.loss.project_dim,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = InfoNCELoss(**kwargs)

    elif model_conf.loss.loss_fn == "RINCELoss":
        kwargs = {
            "temperature": model_conf.loss.temperature,
            "pair_selector": sampling_strategy,
            "projector": model_conf.loss.projector,
            "project_dim": model_conf.loss.project_dim,
            "q": model_conf.loss.q,
            "lam": model_conf.loss.lam,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = RINCELoss(**kwargs)

    elif model_conf.loss.loss_fn == "DecoupledInfoNCELoss":
        kwargs = {
            "temperature": model_conf.loss.temperature,
            "projector": model_conf.loss.projector,
            "project_dim": model_conf.loss.project_dim,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = DecoupledInfoNCELoss(**kwargs)

    elif model_conf.loss.loss_fn == "DecoupledPairwiseInfoNCELoss":
        kwargs = {
            "temperature": model_conf.loss.temperature,
            "pair_selector": sampling_strategy,
            "projector": model_conf.loss.projector,
            "project_dim": model_conf.loss.project_dim,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = DecoupledPairwiseInfoNCELoss(**kwargs)

    elif model_conf.loss.loss_fn == "BinomialDevianceLoss":
        kwargs = {
            "C": model_conf.get("train.C", None),
            "alpha": model_conf.get("train.alpha", None),
            "beta": model_conf.get("train.beta", None),
            "pair_selector": sampling_strategy,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = BinomialDevianceLoss(**kwargs)

    elif model_conf.loss.loss_fn == "TripletLoss":
        kwargs = {
            "margin": model_conf.get("train.margin", None),
            "triplet_selector": sampling_strategy,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = TripletLoss(**kwargs)

    elif model_conf.loss.loss_fn == "HistogramLoss":
        kwargs = {
            "num_steps": model_conf.get("train.num_steps", None),
            "device": torch.device(model_conf["device"]),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = HistogramLoss(**kwargs)

    elif model_conf.loss.loss_fn == "MarginLoss":
        kwargs = {
            "margin": model_conf.get("train.margin", None),
            "beta": model_conf.get("train.beta", None),
            "pair_selector": sampling_strategy,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        loss_fn = MarginLoss(**kwargs)
    else:
        raise AttributeError(f'wrong loss "{model_conf["train.loss"]}"')

    return loss_fn


def outer_cosine_similarity(A, B=None):
    """
    Compute cosine_similarity of Tensors
        A (size(A) = n x d, where n - rows count, d - vector size) and
        B (size(A) = m x d, where m - rows count, d - vector size)
    return matrix C (size n x m), such as C_ij = cosine_similarity(i-th row matrix A, j-th row matrix B)

    if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if B is None:
        B = A

    max_size = 2**32
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)

    if n * m * d <= max_size or m == 1:
        A_norm = torch.div(A.transpose(0, 1), A.norm(dim=1)).transpose(0, 1)
        B_norm = torch.div(B.transpose(0, 1), B.norm(dim=1)).transpose(0, 1)
        return torch.mm(A_norm, B_norm.transpose(0, 1))

    else:
        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_cosine_similarity(A, B[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)
