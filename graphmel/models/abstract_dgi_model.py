import logging
from abc import ABC
from typing import List

import torch
from torch.cuda.amp import autocast
from torch_geometric.loader.neighbor_sampler import EdgeIndex


class AbstractDGIModel(ABC):
    @staticmethod
    def summary_fn(z, *args, **kwargs):
        batch_size = kwargs.get("batch_size")
        if batch_size is not None:
            z = z[:batch_size]
        return torch.sigmoid(z.mean(dim=0))

    @staticmethod
    def corruption_fn(embs, adjs: List[EdgeIndex], *args, **kwargs):
        corrupted_adjs_list = []
        for adj in adjs:
            edge_index = adj.edge_index
            # size = adj.size
            edge_index_src = edge_index[0]
            edge_index_trg = edge_index[1]
            num_edges = len(edge_index_trg)
            perm_trg_nodes = torch.randperm(num_edges)
            corr_edge_index_trg = edge_index_trg[perm_trg_nodes]
            corr_edge_index = torch.stack((edge_index_src, corr_edge_index_trg)).to(edge_index.device)
            corrupted_adj = EdgeIndex(corr_edge_index, adj.e_id, adj.size).to(edge_index.device)
            corrupted_adjs_list.append(corrupted_adj)
        return embs, corrupted_adjs_list


    def graph_encode(self, text_embed_1, text_embed_2, adjs, batch_size, **kwargs):
        pos_graph_embs_1, neg_graph_embs_1, graph_summary_1 = self.dgi(text_embed_1, adjs, batch_size=batch_size,
                                                                       **kwargs)
        pos_graph_embs_2, neg_graph_embs_2, graph_summary_2 = self.dgi(text_embed_2, adjs, batch_size=batch_size,
                                                                       **kwargs)
        pos_graph_embs_1, neg_graph_embs_1 = pos_graph_embs_1[:batch_size], neg_graph_embs_1[:batch_size]
        pos_graph_embs_2, neg_graph_embs_2 = pos_graph_embs_2[:batch_size], neg_graph_embs_2[:batch_size]
        assert pos_graph_embs_1.size()[0] == neg_graph_embs_1.size()[0] == batch_size
        assert pos_graph_embs_2.size()[0] == neg_graph_embs_2.size()[0] == batch_size

        return pos_graph_embs_1, neg_graph_embs_1, graph_summary_1, pos_graph_embs_2, neg_graph_embs_2, graph_summary_2

    def graph_emb(self, text_embed_1, text_embed_2, adjs, batch_size, **kwargs):
        pos_graph_emb_1 = self.graph_encoder(text_embed_1, adjs, batch_size=batch_size, **kwargs)[:batch_size]
        pos_graph_emb_2 = self.graph_encoder(text_embed_2, adjs, batch_size=batch_size, **kwargs)[:batch_size]
        assert pos_graph_emb_1.size()[0] == pos_graph_emb_2.size()[0] == batch_size

        return pos_graph_emb_1, pos_graph_emb_2
