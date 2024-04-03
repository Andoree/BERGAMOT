from typing import List, Dict

import torch
from graphmel.utils.data_utils import node_ids2tokenizer_output
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
import codecs
import logging
import random
from typing import Dict, List, Tuple, Set

from torch.utils.data import Dataset
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast

from graphmel.utils.io import load_node_id2terms_list, load_edges_tuples, load_adjacency_list


class PositivePairNeighborSampler(RawNeighborSampler):
    def __init__(self, pos_pairs_term_1_id_list: List[int], pos_pairs_term_2_id_list: List[int],
                 pos_pairs_concept_ids_list: List[int], term_id2tokenizer_output: Dict,
                 node_id2token_ids_dict, seq_max_length, *args, **kwargs):
        super(PositivePairNeighborSampler, self).__init__(*args, **kwargs)
        self.node_id_to_token_ids_dict = node_id2token_ids_dict
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list

        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.seq_max_length = seq_max_length

    def __len__(self):
        return len(self.pos_pairs_term_1_id_list) // self.batch_size

    def sample(self, batch):
        term_1_ids = [self.pos_pairs_term_1_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.pos_pairs_term_2_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])

        assert len(triplet_concept_ids) == len(term_1_input_ids)
        (batch_size, n_id, adjs) = super(PositivePairNeighborSampler, self).sample(triplet_concept_ids)
        neighbor_node_ids = n_id[batch_size:]

        term_1_neighbor_input_ids, term_1_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id_to_token_ids_dict,
            seq_max_length=self.seq_max_length)
        term_2_neighbor_input_ids, term_2_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id_to_token_ids_dict,
            seq_max_length=self.seq_max_length)

        assert term_1_neighbor_input_ids.size() == term_1_neighbor_att_masks.size() \
               == term_2_neighbor_att_masks.size()
        assert term_2_neighbor_input_ids.size() == term_2_neighbor_att_masks.size()

        term_1_input_ids = torch.cat((term_1_input_ids, term_1_neighbor_input_ids), dim=0)
        term_1_att_masks = torch.cat((term_1_att_masks, term_1_neighbor_att_masks), dim=0)
        term_2_input_ids = torch.cat((term_2_input_ids, term_2_neighbor_input_ids), dim=0)
        term_2_att_masks = torch.cat((term_2_att_masks, term_2_neighbor_att_masks), dim=0)
        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "adjs": adjs, "batch_size": batch_size,
            "concept_ids": n_id  # "concept_ids": triplet_concept_ids
        }
        return batch_dict


class PositiveRelationalNeighborSampler(RawNeighborSampler):
    def __init__(self, pos_pairs_term_1_id_list: List[int], pos_pairs_term_2_id_list: List[int],
                 pos_pairs_concept_ids_list: List[int], term_id2tokenizer_output: Dict,
                 rel_ids, node_id2token_ids_dict, seq_max_length, *args, **kwargs):
        super(PositiveRelationalNeighborSampler, self).__init__(*args, **kwargs)
        self.node_id2token_ids_dict = node_id2token_ids_dict
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list
        self.rel_ids = rel_ids
        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.seq_max_length = seq_max_length

        self.num_edges = self.edge_index.size()[1]

        assert self.num_edges == len(rel_ids)

    def __len__(self):
        return len(self.pos_pairs_term_1_id_list) // self.batch_size

    def sample(self, batch):
        term_1_ids = [self.pos_pairs_term_1_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.pos_pairs_term_2_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])
        assert len(triplet_concept_ids) == len(term_1_input_ids)

        (batch_size, n_id, adjs) = super(PositiveRelationalNeighborSampler, self).sample(triplet_concept_ids)

        neighbor_node_ids = n_id[batch_size:]

        if not isinstance(adjs, list):
            adjs = [adjs, ]
        e_ids_list = [adj.e_id for adj in adjs]
        rel_ids_list = [self.rel_ids[e_ids] for e_ids in e_ids_list]

        term_1_neighbor_input_ids, term_1_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
            seq_max_length=self.seq_max_length)
        term_2_neighbor_input_ids, term_2_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
            seq_max_length=self.seq_max_length)
        assert term_1_neighbor_input_ids.size() == term_1_neighbor_att_masks.size() \
               == term_2_neighbor_att_masks.size()
        assert term_2_neighbor_input_ids.size() == term_2_neighbor_att_masks.size()

        term_1_input_ids = torch.cat((term_1_input_ids, term_1_neighbor_input_ids), dim=0)
        term_1_att_masks = torch.cat((term_1_att_masks, term_1_neighbor_att_masks), dim=0)
        term_2_input_ids = torch.cat((term_2_input_ids, term_2_neighbor_input_ids), dim=0)
        term_2_att_masks = torch.cat((term_2_att_masks, term_2_neighbor_att_masks), dim=0)
        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "adjs": adjs, "batch_size": batch_size,
            "concept_ids": n_id, "rel_ids_list": rel_ids_list,  # "concept_ids": triplet_concept_ids

        }
        return batch_dict


def tokenize_node_terms(node_id_to_terms_dict, tokenizer, max_length: int) -> Dict[int, List[List[int]]]:
    node_id_to_token_ids_dict = {}
    for node_id, terms_list in node_id_to_terms_dict.items():
        node_tokenized_terms = []
        for term in terms_list:
            tokenizer_output = tokenizer.encode_plus(term, max_length=max_length,
                                                     padding="max_length", truncation=True, add_special_tokens=True,
                                                     return_tensors="pt", )
            node_tokenized_terms.append(tokenizer_output)
        node_id_to_token_ids_dict[node_id] = node_tokenized_terms
    return node_id_to_token_ids_dict


def tokenize_node_terms_faster(node_id_to_terms_dict, tokenizer, max_length: int) -> Dict[int, List[List[int]]]:
    terms: List[str] = []
    n_ids = []
    logging.info(f"using faster tokenization")
    for key in sorted(node_id_to_terms_dict.keys()):
        t_list: List[str] = node_id_to_terms_dict[key]
        num_terms = len(t_list)
        terms.extend(t_list)
        n_ids.extend([key, ] * num_terms)
    tokenizer_output = tokenizer.batch_encode_plus(terms, max_length=max_length,
                                                   padding="max_length", truncation=True,
                                                   add_special_tokens=True,
                                                   return_tensors="pt", )
    assert len(tokenizer_output["input_ids"]) == len(n_ids)
    del terms
    node_id_to_token_ids_dict = {}
    for i in tqdm(range(len(n_ids))):
        inp_ids = tokenizer_output["input_ids"][i]
        att_mask = tokenizer_output["attention_mask"][i]
        node_id = n_ids[i]
        if node_id_to_token_ids_dict.get(node_id) is None:
            node_id_to_token_ids_dict[node_id] = []
        node_id_to_token_ids_dict[node_id].append({
            "input_ids": inp_ids,
            "attention_mask": att_mask
        })

    return node_id_to_token_ids_dict


def create_one_hop_adjacency_lists(num_nodes: int, edge_index):
    adjacency_lists = [set() for _ in range(num_nodes)]
    for (node_id_1, node_id_2) in zip(edge_index[0], edge_index[1]):
        adjacency_lists[node_id_1].add(node_id_2)
    return adjacency_lists


def load_data_and_bert_model(train_node2terms_path: str, train_edges_path: str, val_node2terms_path: str,
                             val_edges_path: str, text_encoder_name: str, text_encoder_seq_length: int,
                             drop_relations_info: bool, use_fast: bool = True, do_lower_case=True,
                             no_val=True, tokenization_type="old"):
    train_node_id2terms_dict = load_node_id2terms_list(dict_path=train_node2terms_path, )
    train_edges_tuples = load_edges_tuples(train_edges_path)
    if drop_relations_info:
        train_edges_tuples = [(t[0], t[1]) for t in train_edges_tuples]

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, do_lower_case=do_lower_case, use_fast=use_fast)
    bert_encoder = AutoModel.from_pretrained(text_encoder_name, )

    val_node_id2token_ids_dict = None
    val_edges_tuples = None
    if not no_val:
        val_node_id2terms_dict = load_node_id2terms_list(dict_path=val_node2terms_path, )
        val_edges_tuples = load_edges_tuples(val_edges_path)
        if drop_relations_info:
            val_edges_tuples = [(t[0], t[1]) for t in val_edges_tuples]
        logging.info("Tokenizing val node names")
        val_node_id2token_ids_dict = tokenize_node_terms(val_node_id2terms_dict, tokenizer,
                                                         max_length=text_encoder_seq_length)

    logging.info("Tokenizing training node names")
    if tokenization_type == "old":
        train_node_id2token_ids_dict = tokenize_node_terms(train_node_id2terms_dict, tokenizer,
                                                           max_length=text_encoder_seq_length)
    elif tokenization_type == "faster":
        train_node_id2token_ids_dict = tokenize_node_terms_faster(train_node_id2terms_dict, tokenizer,
                                                                  max_length=text_encoder_seq_length)
    else:
        raise Exception(f"Invalid tokenization_type: {tokenization_type}")

    return bert_encoder, tokenizer, train_node_id2token_ids_dict, train_edges_tuples, val_node_id2token_ids_dict, val_edges_tuples


def convert_edges_tuples_to_oriented_edge_index_with_relations(edges_tuples: List[Tuple[int, int]],
                                                               use_rel_or_rela: str, remove_selfloops=False) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    logging.info("Converting edge tuples to edge index")
    edge_strs_set = set()
    for idx, (node_id_1, node_id_2, rel_id, rela_id) in enumerate(edges_tuples):
        if use_rel_or_rela == "rel":
            pass
        elif use_rel_or_rela == "rela":
            rel_id = rela_id
        else:
            raise ValueError(f"Invalid 'use_rel_or_rela' parameter value: {use_rel_or_rela}")
        if not (remove_selfloops and node_id_1 == node_id_2):
            edge_str = f"{node_id_1}~{node_id_2}~{rel_id}"
            edge_strs_set.add(edge_str)
    edge_index = torch.zeros(size=[2, len(edge_strs_set)], dtype=torch.long)
    edge_rel_ids = torch.zeros(len(edge_strs_set), dtype=torch.long)
    for idx, edge_str in enumerate(edge_strs_set):
        edge_attributes = edge_str.split('~')
        node_id_1 = int(edge_attributes[0])
        node_id_2 = int(edge_attributes[1])
        rel_id = int(edge_attributes[2])

        edge_index[0][idx] = node_id_1
        edge_index[1][idx] = node_id_2
        edge_rel_ids[idx] = rel_id
    logging.info(f"Edge index is created. The size is {edge_index.size()}, there are {edge_index.max()} nodes")

    return edge_index, edge_rel_ids


def load_positive_pairs(triplet_file_path: str) -> Tuple[List[str], List[str], List[int]]:
    term_1_list: List[str] = []
    term_2_list: List[str] = []
    concept_ids: List[int] = []
    with codecs.open(triplet_file_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('||')
            concept_id = int(attrs[0])
            term_1 = attrs[1]
            term_2 = attrs[2]

            concept_ids.append(concept_id)
            term_1_list.append(term_1)
            term_2_list.append(term_2)
    return term_1_list, term_2_list, concept_ids


def map_terms2term_id(term_1_list: List[str], term_2_list: List[str]) -> Tuple[List[int], List[int], Dict[str, int]]:
    unique_terms: Set[str] = set()
    unique_terms.update(term_1_list)
    unique_terms.update(term_2_list)
    term2id = {term: term_id for term_id, term in enumerate(sorted(unique_terms))}

    term_1_ids = [term2id[term] for term in term_1_list]
    term_2_ids = [term2id[term] for term in term_2_list]

    return term_1_ids, term_2_ids, term2id


def create_term_id2tokenizer_output(term2id: Dict[str, int], max_length: int, tokenizer: BertTokenizerFast):
    logging.info("Tokenizing terms....")
    term_id2tok_out = {}
    for term, term_id in term2id.items():
        tok_out = tokenizer.encode_plus(
            term,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        term_id2tok_out[term_id] = tok_out
    logging.info("Finished tokenizing terms....")
    return term_id2tok_out


def convert_edges_tuples_to_edge_index(edges_tuples: List[Tuple[int, int]], remove_selfloops=False) -> torch.Tensor:
    logging.info("Converting edge tuples to edge index")
    # edge_index = torch.zeros(size=[2, len(edges_tuples)], dtype=torch.long)
    edge_strs_set = set()
    for idx, (node_id_1, node_id_2) in enumerate(edges_tuples):
        if node_id_2 < node_id_1:
            node_id_1, node_id_2 = node_id_2, node_id_1
        if not (remove_selfloops and node_id_1 == node_id_2):
            edge_str = f"{node_id_1}~{node_id_2}"
            edge_strs_set.add(edge_str)
    edge_index = torch.zeros(size=[2, len(edge_strs_set) * 2], dtype=torch.long)
    for idx, edge_str in enumerate(edge_strs_set):
        ids = edge_str.split('~')
        node_id_1 = int(ids[0])
        node_id_2 = int(ids[1])
        if not (remove_selfloops and node_id_1 == node_id_2):
            edge_index[0][idx] = node_id_1
            edge_index[1][idx] = node_id_2
            edge_index[0][len(edge_strs_set) + idx] = node_id_2
            edge_index[1][len(edge_strs_set) + idx] = node_id_1

    logging.info(f"Edge index is created. The size is {edge_index.size()}, there are {edge_index.max()} nodes")

    return edge_index
