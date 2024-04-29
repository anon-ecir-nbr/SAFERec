import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

from src.models.transformers import SASRec


class SASRecModel(nn.Module):

    def __init__(self, basket_dim, items_num, basket_vectors=None, n_heads=2, n_layers=2, frequency_max=10):

        super().__init__()

        self.basket_dim = basket_dim
        self.items_num = items_num
        self.relu = nn.ReLU()

        self.transformers = SASRec(
            n_items=self.items_num,
            n_heads=n_heads,
            n_layers=n_layers,
            frequency_max=frequency_max,
            dim=self.basket_dim,
            basket_vectors=basket_vectors,
        )

        self.activation = F.relu
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, basket):
        return self.transformers.basket(basket)


class SAFERec(nn.Module):

    def __init__(
        self,
        num_items,
        basket_vectors,
        dim=64,
        device="cpu",
        n_layers=1,
        n_heads=1,
        frequency_max=10,
        batch_size=256,
    ):

        super().__init__()

        self.action_n = num_items

        self.observation_n = dim

        self.model = SASRecModel(
            basket_dim=dim,
            items_num=self.action_n,
            n_layers=n_layers,
            n_heads=n_heads,
            frequency_max=frequency_max,
            basket_vectors=basket_vectors,
        )

        self.optimizer_p = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )

        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def get_recs(self, basket):

        tensor = torch.from_numpy(np.array(basket)).float()[None, :].to(self.device)

        preds = self.model(tensor)

        return preds.detach().cpu().numpy()[0]

    def handle_sample(self, batch):

        baskets = batch["basket"]
        next_baskets = batch["next_basket"]

        next_baskets_true = next_baskets[:, -1]

        with torch.no_grad():
            tt = next_baskets_true.long()
            tt = self.model.transformers.item_embedding(tt).to_dense()
            nb_vectors = tt.to(self.device)

        preds = self.model(baskets)

        z = F.log_softmax(preds, 1)
        loss = -(z * nb_vectors).sum(1).mean()

        self.optimizer_p.zero_grad(set_to_none=True)

        loss.backward()
        self.optimizer_p.step()
