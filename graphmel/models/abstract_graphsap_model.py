from abc import ABC

import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F


class AbstractGraphSapMetricLearningModel(ABC):

    def calc_text_loss_return_text_embeddings(self, term_1_input_ids, term_1_att_masks, term_2_input_ids,
                                              term_2_att_masks, concept_ids, batch_size):

        text_embed_1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        text_embed_2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        text_loss, hard_pairs = self.calculate_sapbert_loss(text_embed_1[:batch_size], text_embed_2[:batch_size],
                                                            concept_ids[:batch_size], )

        return text_loss, text_embed_1, text_embed_2, hard_pairs

    @autocast()
    def calculate_sapbert_loss(self, emb_1, emb_2, concept_ids, hard_pairs=None):
        text_embed = torch.cat([emb_1, emb_2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)
        if hard_pairs is None:
            if self.use_miner:
                hard_pairs = self.miner(text_embed, labels)
                sapbert_loss = self.loss(text_embed, labels, hard_pairs)
            else:
                sapbert_loss = self.loss(text_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(text_embed, labels, hard_pairs)
        return sapbert_loss, hard_pairs

    @autocast()
    def calculate_intermodal_loss(self, text_embed_1, text_embed_2, graph_embed_1, graph_embed_2, concept_ids,
                                  batch_size, hard_pairs=None):
        intermodal_loss = None
        if self.modality_distance == "sapbert":

            text_graph_embed_1 = torch.cat([text_embed_1[:batch_size], graph_embed_1[:batch_size]], dim=0)
            text_graph_embed_2 = torch.cat([text_embed_2[:batch_size], graph_embed_2[:batch_size]], dim=0)
            concept_ids = concept_ids[:batch_size]
            labels = torch.cat([concept_ids, concept_ids], dim=0)

            if self.use_intermodal_miner:
                intermodal_hard_pairs_1 = self.intermodal_miner(text_graph_embed_1, labels)
                intermodal_hard_pairs_2 = self.intermodal_miner(text_graph_embed_2, labels)
                intermodal_loss_1 = self.intermodal_loss(text_graph_embed_1, labels, intermodal_hard_pairs_1)
                intermodal_loss_2 = self.intermodal_loss(text_graph_embed_2, labels, intermodal_hard_pairs_2)
                intermodal_loss = (intermodal_loss_1 + intermodal_loss_2) / 2
            else:
                intermodal_loss_1 = self.intermodal_loss(text_graph_embed_1, labels, hard_pairs)
                intermodal_loss_2 = self.intermodal_loss(text_graph_embed_2, labels, hard_pairs)
                intermodal_loss = (intermodal_loss_1 + intermodal_loss_2) / 2

        elif self.modality_distance is not None:
            intermodal_loss_1 = self.intermodal_loss(text_embed_1[:batch_size], graph_embed_1[:batch_size])
            intermodal_loss_2 = self.intermodal_loss(text_embed_2[:batch_size], graph_embed_2[:batch_size])
            intermodal_loss = (intermodal_loss_1 + intermodal_loss_2) / 2
        return intermodal_loss

    @autocast()
    def eval_step_loss(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                       batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        text_embed_1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:batch_size, 0]
        text_embed_2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:batch_size, 0]
        labels = torch.cat([concept_ids, concept_ids], dim=0)
        text_loss, hard_pairs = self.calculate_sapbert_loss(text_embed_1, text_embed_2, labels, batch_size)

        return text_loss
