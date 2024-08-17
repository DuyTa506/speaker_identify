from torch import nn
from abc import abstractmethod

import torch
from torch import nn
from cross_entropy_model import FBankNetV2

class TripletLoss(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity()
        self.margin = margin

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings, reduction='mean'):

        # cosine distance is a measure of dissimilarity. The higher the value, more the two vectors are dissimilar
        # it is calculated as (1 - cosine similarity) and ranges between (0,2)

        positive_distance = 1 - self.cosine_similarity(anchor_embeddings, positive_embeddings)
        negative_distance = 1 - self.cosine_similarity(anchor_embeddings, negative_embeddings)

        losses = torch.max(positive_distance - negative_distance + self.margin,torch.full_like(positive_distance, 0))
        if reduction == 'mean':
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class FBankTripletLossNet(FBankNetV2):

    def __init__(self,num_layers, margin):
        super().__init__(num_layers=num_layers)
        self.loss_layer = TripletLoss(margin)

    def forward(self, anchor, positive, negative):
        n = anchor.shape[0]
        anchor_out = self.network(anchor)
        anchor_out = anchor_out.reshape(n, -1)
        anchor_out = self.linear_layer(anchor_out)

        positive_out = self.network(positive)
        positive_out = positive_out.reshape(n, -1)
        positive_out = self.linear_layer(positive_out)

        negative_out = self.network(negative)
        negative_out = negative_out.reshape(n, -1)
        negative_out = self.linear_layer(negative_out)

        return anchor_out, positive_out, negative_out

    def loss(self, anchor, positive, negative, reduction='mean'):
        loss_val = self.loss_layer(anchor, positive, negative, reduction)
        return loss_val