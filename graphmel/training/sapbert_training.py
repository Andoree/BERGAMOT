import itertools
import logging
import os
from torch.cuda.amp import autocast
import torch
from tqdm import tqdm
from graphmel.models.abstract_graphsap_model import AbstractGraphSapMetricLearningModel
from graphmel.utils.io import update_log_file


def graph_sapbert_val_step(model: AbstractGraphSapMetricLearningModel, batch, amp, device):
    term_1_input_ids, term_1_att_masks = batch["term_1_input"]
    term_1_input_ids, term_1_att_masks = term_1_input_ids.to(device), term_1_att_masks.to(device)
    term_2_input_ids, term_2_att_masks = batch["term_2_input"]
    term_2_input_ids, term_2_att_masks = term_2_input_ids.to(device), term_2_att_masks.to(device)

    batch_size = batch["batch_size"]
    concept_ids = batch["concept_ids"].to(device)

    if amp:
        with autocast():
            loss = model.eval_step_loss(term_1_input_ids=term_1_input_ids, term_1_att_masks=term_1_att_masks,
                                        term_2_input_ids=term_2_input_ids, term_2_att_masks=term_2_att_masks,
                                        concept_ids=concept_ids, batch_size=batch_size)
    else:
        loss = model.eval_step_loss(term_1_input_ids=term_1_input_ids, term_1_att_masks=term_1_att_masks,
                                    term_2_input_ids=term_2_input_ids, term_2_att_masks=term_2_att_masks,
                                    concept_ids=concept_ids, batch_size=batch_size)

    return loss


def graph_sapbert_val_epoch(model: AbstractGraphSapMetricLearningModel, val_loader,
                            amp, device, **kwargs):
    model.eval()
    total_loss = 0
    num_steps = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, miniters=len(val_loader), total=len(val_loader)):
            loss = graph_sapbert_val_step(model=model, batch=batch, amp=amp, device=device)
            num_steps += 1
            total_loss += float(loss)
            # wandb.log({"Val loss": loss.item()})
    total_loss /= (num_steps + 1e-9)
    return total_loss


def save_encoder_tokenizer(encoder, tokenizer, save_path):
    try:
        encoder.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logging.info("Model saved in {}".format(save_path))
    except AttributeError as e:
        encoder.module.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logging.info("Model saved in {}".format(save_path))


def train_graph_sapbert_model(model, tokenizer, train_epoch_fn, train_loader, val_loader, chkpnt_path: str,
                              num_epochs: int, learning_rate: float, weight_decay: float, output_dir: str,
                              save_chkpnt_epoch_interval: int, amp: bool, scaler, device: torch.device,
                              save_chkpnts=True, val_epoch_fn=graph_sapbert_val_epoch, save_graph_encoder=False,
                              **kwargs):
    parallel = kwargs["parallel"]
    bert_learning_rate = kwargs.get("bert_learning_rate")
    if chkpnt_path is not None:
        logging.info(f"Successfully loaded checkpoint from: {chkpnt_path}")
        checkpoint = torch.load(chkpnt_path)
        optimizer = checkpoint["optimizer"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
    else:
        start_epoch = 0
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if bert_learning_rate is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            logging.info(f"Using {bert_learning_rate} LR for BERT and {learning_rate} for other parameters")
            non_bert_modules = [module for name, module in model._modules.items() if name != 'bert_encoder']
            non_bert_params = itertools.chain(*(m.parameters() for m in non_bert_modules))
            optimizer = torch.optim.AdamW([{"params": non_bert_params},
                                           {"params": model.bert_encoder.parameters(), "lr": bert_learning_rate}],
                                          lr=learning_rate, weight_decay=0.2)

    log_file_path = os.path.join(output_dir, "training_log.txt")
    train_loss_history = []
    val_loss_history = []
    logging.info("Starting training process....")
    global_num_steps = 0
    for i in range(start_epoch, start_epoch + num_epochs):
        model = model.to(device)
        epoch_train_loss, num_steps = train_epoch_fn(model=model, train_loader=train_loader, device=device,
                                                     optimizer=optimizer, amp=amp, scaler=scaler, **kwargs)
        global_num_steps += num_steps
        log_dict = {"epoch": i + 1, "train loss": epoch_train_loss, }
        if val_loader is not None:
            epoch_val_loss = val_epoch_fn(model=model, val_loader=val_loader, device=device, amp=amp, **kwargs)
            log_dict["val loss"] = epoch_val_loss
            val_loss_history.append(epoch_val_loss)
        logging.info(', '.join((f"{k}: {v}" for k, v in log_dict.items())))

        train_loss_history.append(epoch_train_loss)
        if i % save_chkpnt_epoch_interval == 0:
            checkpoint = {
                'epoch': i + 1,
                'optimizer': optimizer,
            }

            if parallel:
                checkpoint['model_state'] = model.bert_encoder.cpu().module.state_dict()
            else:
                checkpoint['model_state'] = model.bert_encoder.cpu().state_dict()
            if save_graph_encoder:
                checkpoint["graph_encoder"] = model.graph_encoder.cpu().state_dict()

            chkpnt_path = os.path.join(output_dir, f"checkpoint_e_{i + 1}_steps_{global_num_steps}.pth")
            if save_chkpnts:
                torch.save(checkpoint, chkpnt_path)
            encoder_tokenizer_dir = os.path.join(output_dir, f"encoder_e_{i + 1}_steps_{global_num_steps}/")
            if not os.path.exists(encoder_tokenizer_dir):
                save_encoder_tokenizer(model.bert_encoder.cpu(), tokenizer, encoder_tokenizer_dir)

        update_log_file(path=log_file_path, dict_to_log=log_dict)
    checkpoint = {
        'epoch': start_epoch + num_epochs,
        'optimizer': optimizer,
    }
    if parallel:
        checkpoint['model_state'] = model.bert_encoder.cpu().module.state_dict()
    else:
        checkpoint['model_state'] = model.bert_encoder.cpu().state_dict()
    if save_graph_encoder:
        checkpoint["graph_encoder"] = model.graph_encoder.cpu().state_dict()

    chkpnt_path = os.path.join(output_dir, f"checkpoint_e_{start_epoch + num_epochs}_steps_{global_num_steps}.pth")
    if save_chkpnts:
        torch.save(checkpoint, chkpnt_path)

    encoder_tokenizer_dir = os.path.join(output_dir, f"encoder_e_{start_epoch + num_epochs}_steps_{global_num_steps}/")
    if not os.path.exists(encoder_tokenizer_dir):
        save_encoder_tokenizer(model.bert_encoder.cpu(), tokenizer, encoder_tokenizer_dir)
