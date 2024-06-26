#!/usr/bin/env python
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from tqdm import tqdm

from graphmel.data.dataset import PositivePairNeighborSampler, PositiveRelationalNeighborSampler
from graphmel.models.rgcn_dgi_sapbert import RGCNDGISapMetricLearning
from graphmel.training.sapbert_training import train_graph_sapbert_model
from graphmel.data.dataset import load_positive_pairs, map_terms2term_id, \
    create_term_id2tokenizer_output, load_data_and_bert_model, \
    convert_edges_tuples_to_oriented_edge_index_with_relations
from graphmel.utils.io import save_dict, load_dict

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert train')

    # Required
    # parser.add_argument('--model_dir',
    #                     help='Directory for pretrained model')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='training set directory')
    # parser.add_argument('--val_dir', type=str, required=False,
    #                     help='Validation set directory')
    parser.add_argument('--validate', action="store_true",
                        help='whether the validation of each epoch is required')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')

    # RGCN encoder configuration
    parser.add_argument('--rgcn_num_hidden_channels', type=int)
    parser.add_argument('--rgcn_num_bases', type=int)
    parser.add_argument('--rgcn_num_blocks', type=int)
    parser.add_argument('--use_rel_or_rela', type=str, choices=['rel', 'rela', ])
    parser.add_argument('--rgcn_use_fast_conv', action="store_true")
    parser.add_argument('--rgcn_num_neighbors', type=int, nargs='+')
    parser.add_argument('--rgcn_num_outer_layers', type=int, )
    parser.add_argument('--rgcn_num_inner_layers', type=int, )
    parser.add_argument('--rgcn_dropout_p', type=float, )
    parser.add_argument('--dgi_loss_weight', type=float)
    parser.add_argument('--remove_selfloops', action="store_true")
    parser.add_argument('--text_loss_weight', type=float, required=False, default=1.0)
    parser.add_argument('--graph_loss_weight', type=float, )
    parser.add_argument('--intermodal_loss_weight', type=float, required=False)
    parser.add_argument('--use_intermodal_miner', action="store_true")
    parser.add_argument('--intermodal_miner_margin', default=0.2, type=float, required=False)
    parser.add_argument('--modality_distance', type=str, required=False, choices=(None, "sapbert", "cosine", "MSE"))

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--use_cuda', action="store_true")
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--batch_size',
                        help='train batch size',
                        default=240, type=int)
    parser.add_argument('--num_epochs',
                        help='epoch to train',
                        default=3, type=int)
    parser.add_argument('--amp', action="store_true",
                        help="automatic mixed precision training")
    parser.add_argument('--parallel', action="store_true")
    parser.add_argument('--random_seed',
                        help='',
                        default=1996, type=int)
    parser.add_argument('--loss',
                        help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}",
                        default="ms_loss")
    parser.add_argument('--use_miner', action="store_true")
    parser.add_argument('--miner_margin', default=0.2, type=float)
    parser.add_argument('--type_of_triplets', default="all", type=str)
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_all_tok}")

    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument('--save_every_N_epoch', type=int, default=1)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)

    args = parser.parse_args()
    return args


def rgcn_dgi_sapbert_train_step(model: RGCNDGISapMetricLearning, batch, amp, device):
    term_1_input_ids, term_1_att_masks = batch["term_1_input"]
    term_1_input_ids, term_1_att_masks = term_1_input_ids.to(device), term_1_att_masks.to(device)
    term_2_input_ids, term_2_att_masks = batch["term_2_input"]
    term_2_input_ids, term_2_att_masks = term_2_input_ids.to(device), term_2_att_masks.to(device)
    adjs = batch["adjs"]
    rel_types_list = batch["rel_ids_list"]
    rel_types_list = [rel_types.to(device) for rel_types in rel_types_list]

    adjs = [adj.to(device) for adj in adjs]

    batch_size = batch["batch_size"]
    concept_ids = batch["concept_ids"].to(device)

    if amp:
        with autocast():
            sapbert_loss, graph_loss, dgi_loss, intermodal_loss = model(term_1_input_ids=term_1_input_ids,
                                                            term_1_att_masks=term_1_att_masks,
                                                            term_2_input_ids=term_2_input_ids,
                                                            term_2_att_masks=term_2_att_masks,
                                                            concept_ids=concept_ids, adjs=adjs,
                                                            rel_types=rel_types_list, batch_size=batch_size)
    else:
        sapbert_loss, graph_loss, dgi_loss, intermodal_loss = model(term_1_input_ids=term_1_input_ids,
                                                        term_1_att_masks=term_1_att_masks,
                                                        term_2_input_ids=term_2_input_ids,
                                                        term_2_att_masks=term_2_att_masks,
                                                        concept_ids=concept_ids, adjs=adjs, rel_types=rel_types_list,
                                                        batch_size=batch_size)
    # logging.info(f"Train loss: {float(loss)}")
    return sapbert_loss, graph_loss, dgi_loss, intermodal_loss


def train_rgcn_dgi_sapbert(model: RGCNDGISapMetricLearning, train_loader: PositivePairNeighborSampler,
                           optimizer: torch.optim.Optimizer, scaler, amp, device, **kwargs):
    model.train()
    losses_dict = {"total": 0, "sapbert": 0, "graph": 0, "dgi": 0, "intermodal": 0}
    num_steps = 0
    pbar = tqdm(train_loader, miniters=len(train_loader) // 100, total=len(train_loader))
    for batch in pbar:
        optimizer.zero_grad()
        sapbert_loss, graph_loss, dgi_loss, intermodal_loss = rgcn_dgi_sapbert_train_step(model=model, batch=batch,
                                                                                          amp=amp, device=device)
        sapbert_loss = sapbert_loss * model.sapbert_loss_weight
        graph_loss = graph_loss * model.graph_loss_weight
        dgi_loss = dgi_loss * model.dgi_loss_weight
        if intermodal_loss is not None:
            intermodal_loss = intermodal_loss * model.intermodal_loss_weight
            loss = sapbert_loss + graph_loss + dgi_loss + intermodal_loss
        else:
            loss = sapbert_loss + graph_loss + dgi_loss
            intermodal_loss = -1.
        pbar.set_description(f"L: {float(loss):.5f} ({float(sapbert_loss):.5f} + {float(graph_loss):.5f} + "
                             f"{float(dgi_loss):.5f} + {float(intermodal_loss):.5f})")
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        num_steps += 1
        losses_dict["total"] += float(loss)
        losses_dict["sapbert"] += float(sapbert_loss)
        losses_dict["graph"] += float(graph_loss)
        losses_dict["dgi"] += float(dgi_loss)
        losses_dict["intermodal"] += float(intermodal_loss)

    losses_dict = {key: lo / (num_steps + 1e-9) for key, lo in losses_dict.items()}

    return losses_dict["total"], num_steps


def main(args):
    print(args)
    output_dir = args.output_dir
    assert len(args.rgcn_num_neighbors) == args.rgcn_num_outer_layers
    conv_type = "fast_rgcn_conv" if args.rgcn_use_fast_conv else "rgcn_conv"
    output_subdir = f"dgi_{args.dgi_loss_weight}_rgcn_{args.rgcn_num_outer_layers}_{args.rgcn_num_inner_layers}" \
                    f"_{args.rgcn_num_neighbors}_text_{args.text_loss_weight}_remove_loops_{args.remove_selfloops}" \
                    f"graph_loss_{args.graph_loss_weight}_intermodal_{args.modality_distance}_{args.intermodal_loss_weight}" \
                    f"_{args.rgcn_dropout_p}_{args.rgcn_num_hidden_channels}--{args.rgcn_num_bases}-" \
                    f"{args.rgcn_num_blocks}_{args.use_rel_or_rela}_intermodal_miner_{args.use_intermodal_miner}" \
                    f"_{args.intermodal_miner_margin}" \
                    f"_lr_{args.learning_rate}_b_{args.batch_size}_{conv_type}"
    output_dir = os.path.join(output_dir, output_subdir)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    modality_distance = args.modality_distance
    if modality_distance == "None":
        modality_distance = None
    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=vars(args), )
    torch.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.random.manual_seed(args.random_seed)
    torch.cuda.random.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True

    node2terms_path = os.path.join(args.train_dir, "node_id2terms_list")
    edges_path = os.path.join(args.train_dir, "edges")
    rel2id_path = os.path.join(args.train_dir, "rel2id")
    rela2id_path = os.path.join(args.train_dir, "rela2id")

    bert_encoder, bert_tokenizer, node_id2token_ids_dict, edges_tuples, _, _ = \
        load_data_and_bert_model(train_node2terms_path=node2terms_path,
                                 train_edges_path=edges_path, use_fast=True, do_lower_case=True,
                                 val_node2terms_path=node2terms_path,
                                 val_edges_path=edges_path, text_encoder_name=args.text_encoder,
                                 text_encoder_seq_length=args.max_length, drop_relations_info=False,
                                 no_val=True)

    del _

    # rel2id = load_dict(rel2id_path)
    # rela2id = load_dict(rela2id_path)
    rel2id = {rel: int(i) for rel, i in load_dict(rel2id_path).items()}
    rela2id = {rela: int(i) for rela, i in load_dict(rela2id_path).items()}

    if args.use_rel_or_rela == "rel":
        num_relations = len(rel2id.keys())
    elif args.use_rel_or_rela == "rela":
        num_relations = len(rela2id.keys())
    else:
        raise ValueError(f"Invalid 'use_rel_or_rela' parameter: {args.use_rel_or_rela}")

    edge_index, edge_rel_ids = \
        convert_edges_tuples_to_oriented_edge_index_with_relations(edges_tuples, args.use_rel_or_rela,
                                                                   remove_selfloops=args.remove_selfloops)
    assert edge_index.size()[1] == len(edge_rel_ids)

    num_edges = edge_index.size()[1]
    num_nodes = len(set(node_id2token_ids_dict.keys()))

    del edges_tuples

    train_positive_pairs_path = os.path.join(args.train_dir, f"train_pos_pairs")
    train_pos_pairs_term_1_list, train_pos_pairs_term_2_list, train_pos_pairs_concept_ids = \
        load_positive_pairs(train_positive_pairs_path)
    train_pos_pairs_term_1_id_list, train_pos_pairs_term_2_id_list, train_term2id = map_terms2term_id(
        term_1_list=train_pos_pairs_term_1_list, term_2_list=train_pos_pairs_term_2_list)
    logging.info(f"There are {len(train_pos_pairs_term_1_id_list)} positive training pairs")
    train_term_id2tok_out = create_term_id2tokenizer_output(term2id=train_term2id, max_length=args.max_length,
                                                            tokenizer=bert_tokenizer)
    del train_pos_pairs_term_1_list
    del train_pos_pairs_term_2_list
    logging.info(f"There are {num_nodes} nodes and {num_edges} edges in graph.")
    train_num_pos_pairs = len(train_pos_pairs_term_1_id_list)
    train_pos_pairs_idx = torch.LongTensor(range(train_num_pos_pairs))
    train_pos_pair_sampler = PositiveRelationalNeighborSampler(pos_pairs_term_1_id_list=train_pos_pairs_term_1_id_list,
                                                               pos_pairs_term_2_id_list=train_pos_pairs_term_2_id_list,
                                                               pos_pairs_concept_ids_list=train_pos_pairs_concept_ids,
                                                               sizes=args.rgcn_num_neighbors, edge_index=edge_index,
                                                               term_id2tokenizer_output=train_term_id2tok_out,
                                                               rel_ids=edge_rel_ids, node_idx=train_pos_pairs_idx,
                                                               node_id2token_ids_dict=node_id2token_ids_dict,
                                                               seq_max_length=args.max_length,
                                                               batch_size=args.batch_size,
                                                               num_workers=args.dataloader_num_workers, shuffle=True, )

    val_pos_pair_sampler = None
    if args.validate:
        val_positive_pairs_path = os.path.join(args.train_dir, f"val_pos_pairs")
        val_pos_pairs_term_1_list, val_pos_pairs_term_2_list, val_pos_pairs_concept_ids = \
            load_positive_pairs(val_positive_pairs_path)
        val_pos_pairs_term_1_id_list, val_pos_pairs_term_2_id_list, val_term2id = map_terms2term_id(
            term_1_list=val_pos_pairs_term_1_list, term_2_list=val_pos_pairs_term_2_list)
        logging.info(f"There are {len(val_pos_pairs_term_1_id_list)} positive validation pairs")
        del val_pos_pairs_term_1_list
        del val_pos_pairs_term_2_list
        val_term_id2tok_out = create_term_id2tokenizer_output(term2id=val_term2id, max_length=args.max_length,
                                                              tokenizer=bert_tokenizer)
        val_num_pos_pairs = len(val_pos_pairs_term_1_id_list)
        val_pos_pairs_idx = torch.LongTensor(range(val_num_pos_pairs))
        val_pos_pair_sampler = PositiveRelationalNeighborSampler(
            pos_pairs_term_1_id_list=val_pos_pairs_term_1_id_list,
            pos_pairs_term_2_id_list=val_pos_pairs_term_2_id_list,
            pos_pairs_concept_ids_list=val_pos_pairs_concept_ids,
            sizes=args.rgcn_num_neighbors, edge_index=edge_index,
            term_id2tokenizer_output=val_term_id2tok_out, node_idx=val_pos_pairs_idx,
            rel_ids=edge_rel_ids, node_id2token_ids_dict=node_id2token_ids_dict,
            seq_max_length=args.max_length, batch_size=args.batch_size,
            num_workers=args.dataloader_num_workers, shuffle=True, )
    device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    model = RGCNDGISapMetricLearning(bert_encoder, num_rgcn_channels=args.rgcn_num_hidden_channels,
                                     dgi_loss_weight=args.dgi_loss_weight, rgcn_dropout_p=args.rgcn_dropout_p,
                                     num_relations=num_relations, num_bases=args.rgcn_num_bases,
                                     num_blocks=args.rgcn_num_blocks, use_fast_conv=args.rgcn_use_fast_conv,
                                     num_outer_rgcn_layers=args.rgcn_num_outer_layers,
                                     num_inner_rgcn_layers=args.rgcn_num_inner_layers ,use_cuda=args.use_cuda,
                                     sapbert_loss_weight=args.text_loss_weight,
                                     graph_loss_weight=args.graph_loss_weight, loss=args.loss,
                                     modality_distance=modality_distance,
                                     intermodal_loss_weight=args.intermodal_loss_weight,
                                     multigpu_flag=args.parallel, use_miner=args.use_miner,
                                     miner_margin=args.miner_margin,
                                     use_intermodal_miner=args.use_intermodal_miner,
                                     intermodal_miner_margin=args.intermodal_miner_margin,
                                     type_of_triplets=args.type_of_triplets, agg_mode=args.agg_mode).to(device)
    start = time.time()
    train_graph_sapbert_model(model=model, tokenizer=bert_tokenizer, train_epoch_fn=train_rgcn_dgi_sapbert,
                              train_loader=train_pos_pair_sampler,
                              val_loader=val_pos_pair_sampler, parallel=args.parallel,
                              learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                              num_epochs=args.num_epochs, output_dir=output_dir,
                              save_chkpnt_epoch_interval=args.save_every_N_epoch,
                              amp=args.amp, scaler=scaler, device=device, chkpnt_path=args.model_checkpoint_path)
    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    logging.info(f"Training Time took {training_hour} hours {training_minute} minutes {training_second} seconds")


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
