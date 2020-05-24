#Code is modified from following site :https://github.com/google/tirg


import numpy as np
import torch
import torchvision


def pairwise_distances(x, y=None):

  x_norm = (x**2).sum(1).view(-1, 1)
  if y is not None:
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
  else:
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
  # Ensure diagonal is zero if x=y
  # if y is None:
  #     dist = dist - torch.diag(dist.diag)
  return torch.clamp(dist, 0.0, np.inf)






class TripletLoss(torch.nn.Module):
  """Class for the triplet loss."""
  def __init__(self, pre_layer=None):
    super(TripletLoss, self).__init__()
    self.pre_layer = pre_layer

  def forward(self, x, triplets):
    if self.pre_layer is not None:
      x = self.pre_layer(x)
    

    #modifications
    self.triplets = triplets
    self.triplet_count = len(triplets)
    #self.distances = pairwise_distances(x).detach().cpu().numpy()
    self.distances = pairwise_distances(x)
    loss = 0.0
    triplet_count = 0.0
    correct_count = 0.0
    for i, j, k in self.triplets:
      w = 1.0
      triplet_count += w
      loss += w * torch.log(1 +torch.exp(self.distances[i, j] - self.distances[i, k]))
      if self.distances[i, j] < self.distances[i, k]:
        correct_count += 1

    loss /= triplet_count
    #print("again loss",loss)
    return loss
    #end of modifications

    return loss


class NormalizationLayer(torch.nn.Module):
  """Class for normalization layer."""
  def __init__(self, normalize_scale=1.0, learn_scale=True):
    super(NormalizationLayer, self).__init__()
    self.norm_s = float(normalize_scale)
    if learn_scale:
      self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

  def forward(self, x):
    features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
    return features
