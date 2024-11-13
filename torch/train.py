import gc
from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn

import evaluate

from utils import *

# ========================================================================================================================================================== #

def train_one_epoch(model,
                    accelerator, 
                    data_loader,
                    optimizer,
                    scheduler,
                    step,
                    mode = "train",
                    logging = True,
                    ):

    model.train()
    total_loss = 0

    bar = tqdm(data_loader, total= len(data_loader))
    for batch in bar:
        with accelerator.accumulate(model):
            
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs.loss
                
            # loss_gen_all.backward()
            accelerator.backward(loss)
            
        total_loss += loss.detach().float()

        # Gradient-Clipping | source: https://velog.io/@seven7724/Transformer-계열의-훈련-Tricks
        if accelerator.sync_gradients:
            max_grad_norm = 1
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()

            # Zero the gradients
            optimizer.zero_grad()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()
        
        
        log_metrics = {f"{mode}/loss": loss.item() }
        if logging:
            accelerator.log(log_metrics, step=step)
        
        step += 1
            

    # Train Epoch Loss
    train_epoch_loss = total_loss / len(data_loader)

    # epoch_metric
    log_metrics_epoch = {f"{mode}/total_loss": train_epoch_loss}
    
    if torch.cuda.is_available() and condition_for_memory_cleanup():
        torch.cuda.empty_cache()  # 필요할 때만 GPU 메모리 해제
        gc.collect()

    return model, train_epoch_loss, step, log_metrics_epoch 

# ========================================================================================================================================================== #

@torch.inference_mode()
def valid_one_epoch(model,
                    accelerator, 
                    data_loader,
                    step,
                    mode = "valid",
                    logging = True,
                    ):

    model.eval()
    total_loss = 0

    bar = tqdm(data_loader, total= len(data_loader))
    with torch.no_grad():
        for batch in bar:
        
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.detach().float()
            
            log_metrics = {f"{mode}/loss": loss.item() }
            
            if logging:
                accelerator.log(log_metrics, step=step)
            
    # Valid Epoch Loss
    valid_epoch_loss = total_loss / len(data_loader)

    # epoch_metric
    log_metrics_epoch = {f"{mode}/total_loss": valid_epoch_loss}
    
    if torch.cuda.is_available() and condition_for_memory_cleanup():
        torch.cuda.empty_cache()  # 필요할 때만 GPU 메모리 해제
        gc.collect()

    return model, valid_epoch_loss, step, log_metrics_epoch

# ========================================================================================================================================================== #

@torch.inference_mode()
def evaluate_func(  model, 
                    accelerator, 
                    data_loader,
                    tokenizer
                ):
    
    # import evaluate
    metric = evaluate.load('cer')
    
    # processor.tokenizer
    # or tokenizer
    
    model.eval()
    eval_preds, eval_labels = [], []
    
    bar = tqdm(data_loader, total= len(data_loader))
    with torch.no_grad():
        for batch in bar:

            outputs = accelerator.unwrap_model(model).generate(**batch, 
                                                                # forced_decoder_ids=forced_decoder_ids,
                                                                # max_length=512, 
                                                                # num_beams = 3, 
                                                                # early_stopping=True,
                                                                use_cache=True
                                                                )
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            outputs = accelerator.gather_for_metrics(outputs)
            eval_preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

            # WER, ACC
            labels = batch['labels'].detach().cpu().numpy()
            labels[labels == -100] = tokenizer.pad_token_id
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            eval_labels += labels

    # CER SCORE
    cer_score = 100 * metric.compute(predictions=eval_preds, references=eval_labels)
    
    if torch.cuda.is_available() and condition_for_memory_cleanup():
        torch.cuda.empty_cache()  # 필요할 때만 GPU 메모리 해제
        gc.collect()
    
    return {'cer': cer_score}


    # evaluate metric function
    # def compute_metrics(pred):
    #     # import evaluate
    #     metric = evaluate.load('cer')

    #     pred_ids = pred.predictions
    #     label_ids = pred.label_ids

    #     # pad_token을 -100으로 치환
    #     label_ids[label_ids == -100] = tokenizer.pad_token_id

    #     # metrics 계산 시 special token들을 빼고 계산하도록 설정
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    #     cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    #     return {"cer": cer}


