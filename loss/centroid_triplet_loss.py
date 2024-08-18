import torch
import torch.nn as nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class CentroidLoss(object):
    def __init__(self, margin=None):
        self.margin = margin  #
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        n = global_feat.size(0)
        k = len(torch.unique(labels))
        unique_labels = torch.unique(labels)
        label_map = dict()

        for i, label in enumerate(unique_labels):
            label_map[label.item()] = i
        label_new = torch.zeros_like(labels)
        for i in range(len(labels)):
            label_new[i] = label_map[labels[i].item()]

        centroids = torch.zeros(k, global_feat.size(1)).to(global_feat.device)
        for i in range(k):
            indices = (label_new == i).nonzero().squeeze(1)
            centroids[i] = torch.mean(global_feat[indices], dim=0)

        distance = torch.zeros(n, k).to(global_feat.device)
        for i in range(n):
            for j in range(k):
                distance[i][j] = torch.norm(global_feat[i] - centroids[j], p=2)

        distance_ap, distance_an = [], []
        for i in range(n):
            label = label_new[i].item()
            distance_ap.append(distance[i][label].unsqueeze(0))
            distance_an.append(distance[i][distance[i] != distance[i][label]].min().unsqueeze(0))

        distance_ap = torch.cat(distance_ap)
        distance_an = torch.cat(distance_an)

        y = torch.ones_like(distance_an)
        if self.margin is not None:
            loss = self.ranking_loss(distance_ap, distance_an, y)
        else:
            loss = self.ranking_loss(distance_an - distance_ap, y)
        return loss, distance_ap, distance_an
