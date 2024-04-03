import logging

import torch

EPS = 1e-15
from torch_geometric.nn import DeepGraphInfomax


class Float32DeepGraphInfomax(DeepGraphInfomax):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """

    def __init__(self, hidden_channels, encoder, summary, corruption):
        super(Float32DeepGraphInfomax, self).__init__(hidden_channels=hidden_channels,
                                                      encoder=encoder, summary=summary, corruption=corruption)

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        This method is the modified version of the implementation from PyTorch Geometric but it calculates 'value'
        variable using torch.float32 datatype instead of torch.float16. Without this fix the model may fail to compute
        DGI loss.

        Args:
            z (Tensor): The latent space.
            summary (Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary)).float()
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(*args, **kwargs)
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor,)
        neg_z = self.encoder(*cor, **kwargs)
        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary

