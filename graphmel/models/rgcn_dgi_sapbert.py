import logging

import torch
from pytorch_metric_learning import miners, losses
from torch import nn as nn
from torch.cuda.amp import autocast

from graphmel.models.abstract_dgi_model import AbstractDGIModel
from graphmel.models.abstract_graphsap_model import AbstractGraphSapMetricLearningModel
from graphmel.models.modules import ModalityDistanceLoss, RGCNEncoder
from graphmel.models.dgi import Float32DeepGraphInfomax


class RGCNDGISapMetricLearning(nn.Module, AbstractGraphSapMetricLearningModel, AbstractDGIModel):
    def __init__(self, bert_encoder, num_outer_rgcn_layers: int, num_inner_rgcn_layers: int, num_rgcn_channels: int,
                 rgcn_dropout_p: float, graph_loss_weight: float, dgi_loss_weight: float, intermodal_loss_weight: float,
                 num_relations: int, num_bases: int, num_blocks: int, use_fast_conv: bool, use_cuda, loss,
                 multigpu_flag, use_intermodal_miner=True, intermodal_miner_margin=0.2, use_miner=True,
                 miner_margin=0.2, type_of_triplets="all", agg_mode="cls", modality_distance=None,
                 sapbert_loss_weight: float = 1.0):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))

        super(RGCNDGISapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.use_intermodal_miner = use_intermodal_miner
        self.agg_mode = agg_mode
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder

        self.sapbert_loss_weight = sapbert_loss_weight
        self.graph_loss_weight = graph_loss_weight
        self.dgi_loss_weight = dgi_loss_weight
        self.intermodal_loss_weight = intermodal_loss_weight
        self.modality_distance = modality_distance
        if modality_distance == "sapbert":
            if self.use_intermodal_miner:
                self.intermodal_miner = miners.TripletMarginMiner(margin=intermodal_miner_margin,
                                                                  type_of_triplets=type_of_triplets)
            else:
                self.intermodal_miner = None
            self.intermodal_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

        elif modality_distance is not None:
            self.intermodal_loss = ModalityDistanceLoss(dist_name=modality_distance)

        self.rgcn_conv = RGCNEncoder(in_channels=self.bert_hidden_dim, num_outer_layers=num_outer_rgcn_layers,
                                     num_inner_layers=num_inner_rgcn_layers, num_hidden_channels=num_rgcn_channels,
                                     dropout_p=rgcn_dropout_p, num_relations=num_relations, num_blocks=num_blocks,
                                     num_bases=num_bases, use_fast_conv=use_fast_conv,
                                     set_out_input_dim_equal=True)

        self.dgi = Float32DeepGraphInfomax(
            hidden_channels=self.bert_hidden_dim, encoder=self.rgcn_conv,
            summary=self.summary_fn, corruption=self.corruption_fn)

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # 1,2,3; 40,50,60
        else:
            raise ValueError(f"Invalid loss string: {self.loss}")
        logging.info(f"Using miner: {self.miner}")
        logging.info(f"Using loss function: {self.loss}")

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                adjs, rel_types, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        text_loss, text_embed_1, text_embed_2, hard_pairs = \
            self.calc_text_loss_return_text_embeddings(term_1_input_ids, term_1_att_masks,
                                                       term_2_input_ids, term_2_att_masks, concept_ids, batch_size)

        pos_graph_embs_1, neg_graph_embs_1, graph_summary_1, pos_graph_embs_2, neg_graph_embs_2, graph_summary_2 = \
            self.graph_encode(text_embed_1, text_embed_2, adjs=adjs, rel_types=rel_types, batch_size=batch_size)

        dgi_loss_1 = self.dgi.loss(pos_graph_embs_1, neg_graph_embs_1, graph_summary_1)
        dgi_loss_2 = self.dgi.loss(pos_graph_embs_2, neg_graph_embs_2, graph_summary_2)

        graph_loss, hard_pairs = self.calculate_sapbert_loss(pos_graph_embs_1[:batch_size], pos_graph_embs_2[:batch_size],
                                                 concept_ids[:batch_size], hard_pairs=hard_pairs)

        intermodal_loss = self.calculate_intermodal_loss(text_embed_1, text_embed_2, pos_graph_embs_1, pos_graph_embs_2,
                                                         concept_ids, batch_size, hard_pairs=hard_pairs)

        return text_loss, graph_loss, (dgi_loss_1 + dgi_loss_2) / 2, intermodal_loss
