import os
import gc
import json

import wandb
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta

# import librosa
import IPython.display as ipd
# import soundfile as sf
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio

# Huggingface: Model Download
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer

# Accelerate
from accelerate import Accelerator
# from accelerate import InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed 

# others
from torch_dataset import *
from trainer import *
from model import *
from utils import *

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
s_ = Fore.CYAN
y_ = Fore.YELLOW
r_ = Fore.RED
g_= Fore.GREEN
sr_ = Style.RESET_ALL
m_ = Fore.MAGENTA 

# ========================================================================================================================================================== #

def main(args):
    
    # HParams
    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hps = HParams(**config)
    print(f"{b_} config loaded {sr_}")
    print()
    
    # Accelerator
    # from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
	# timeout Issue
    # ipg_handler = InitProcessGroupKwargs(
    #     timeout=timedelta(seconds=5400)
    #         )
    
    accelerator = Accelerator(    
                log_with=hps.accelerator.log_with, 
                mixed_precision=hps.accelerator.mixed_precision,
                gradient_accumulation_steps = hps.accelerator.grad_acc_step, 
                kwargs_handlers=[kwargs] 
                # kwargs_handlers=[ipg_handler]
                )
    
    model_name = hps.train.model_name
    lang = hps.train.lang
    print(f"{r_} model_name: {model_name}, lang: {lang}, task: {hps.train.task} {sr_}")
    # model_name = "openai/whisper-small"
    # lang = "Korean"

    batch_size = hps.train.train_batch_size, 
    gradient_accumulation_steps = hps.accelerator.grad_acc_step
    print(f"{b_} Batch Size: {batch_size}, Gradient_Accumulation_Steps: {gradient_accumulation_steps} {sr_}")
    print()

    # Set Seed
    set_seed(hps.train.seed)
    print(f"{y_} Set Seed! {sr_}")
    print()
    
    # DataFrame Load:
    df = pd.read_csv(hps.train.dataset_path)
    print(df.shape)
    print(f"{g_} Dataframe(=transciption, Audio Info) loaded! {sr_}")
    print()
    
    # Processor and others download:
    # 파인튜닝을 진행하고자 하는 모델의 feature extractor를 로드
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    # 파인튜닝을 진행하고자 하는 모델의 tokenizer를 로드
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language = lang, task=hps.train.task)

    # All we need is Processor
    processor = WhisperProcessor.from_pretrained(model_name, language = lang, task=hps.train.task)
    # task(=hps.train.task): "transcribe"
    
    print(f"{g_} feature_extractor, processor, tokenizer loaded {sr_}")
    
    # DataLoader
    train_loader, valid_loader = prepare_loaders(df= df, 
                                                train_ratio = hps.train.train_ratio, 
                                                processor = processor,
                                                batch_size = (hps.train.train_batch_size, hps.train.valid_batch_size)
                                                )
    print(f"{m_} DataLoaders loaded {sr_}")

    # Model Downloaded
    model = get_whisper_model(model_name = model_name, device = None)
    
    # Model Downloaded
    print(f"{b_} Model Downloaded {sr_}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr =  hps.optimizer.lr)
    print(f"{r_} Optimizer Defined {sr_}")
    
    # Scheduler
    steps_per_epoch = int( df.shape[0] * hps.train.train_ratio) // hps.train.train_batch_size
    print(f"{s_} steps_per_epoch: {steps_per_epoch} {sr_}")
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = hps.optimizer.lr,
                                                    steps_per_epoch = int(steps_per_epoch),
                                                    epochs = hps.train.n_epochs
                                                    )
    print(f"{s_} Scheduler Defined {sr_}")
    
    # Accelerate prepare: to GPU
    model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, valid_loader)
    
    # model cuda? Check!
    print(f"{y_} Model is on GPU?: {next(model.parameters()).is_cuda} {sr_}")

    # ================ Accelerator & wandb ================ #

    # you should log in (wandb) in cli 
    wandb_config= {
        'dataset': "저음질 전화 음성인식 데이터",
        'model': hps.train.project_name,
        "n_epochs": hps.train.n_epochs,
        'batch_size': hps.train.train_batch_size,
        'total_step': int(hps.train.n_epochs * len(train_loader)),
    }
    
    ## wandb init with accelerator
    accelerator.init_trackers(
        project_name= hps.train.project_name,
        config= wandb_config,
        init_kwargs={
            "wandb": {
                "job_type": 'train',
                "tags": [hps.train.project_name, "Korean"],
                "name": hps.train.try_name,
                }
            },
        )
    accelerator.print(f"{y_}Accelerator, wandb are gonna be initialized{sr_}")
    print()

# ================ Run Training ================ #
    
    step = 0
    valid_loss = 0
    lowest_loss = np.inf
    lowest_epoch = 0

    n_epochs = hps.train.n_epochs
    eval_iter = hps.train.eval_iter
    print_iter = hps.train.print_iter
    
    print(f"{y_} PRJ: {hps.train.project_name} {sr_}")
    print(f"{g_} TRY: {hps.train.try_name} {sr_}")
    print(f"{b_}============ Start Running ============{sr_}")
    for epoch in trange(n_epochs, desc = 'Epoch'):

        # ================ train(=train_one_epoch) ================ #
        model, train_epoch_loss, step, train_log_metrics_epoch = train_one_epoch( model, 
                                                                            accelerator,
                                                                            train_loader,
                                                                            optimizer,
                                                                            scheduler,
                                                                            step, 
                                                                            mode ="train",
                                                                            logging = True,
									    gradient_accumulation_steps = gradient_accumulation_steps
                                                                            )
        accelerator.print(f"Train Epoch: {epoch} Finished")
        # train-epoch_metric
        train_log_metrics_epoch["epoch"] = epoch
        accelerator.log(train_log_metrics_epoch)
        accelerator.print(f"Train Epoch: {epoch} Logging Finished")
        print()
        
        
        # ================ valid(=valid_one_epoch) ================ #
        model, valid_epoch_loss, step, valid_log_metrics_epoch = valid_one_epoch( model, 
                                                                            accelerator,
                                                                            valid_loader,
                                                                            step, 
                                                                            mode ="valid",
                                                                            logging = True,
                                                                            )
        accelerator.print(f"Valid Epoch: {epoch} Finished")
        # valid-epoch_metric
        valid_log_metrics_epoch["epoch"] = epoch
        accelerator.log(valid_log_metrics_epoch)
        accelerator.print(f"Valid Epoch: {epoch} Logging Finished")
        print()

        # 2 epochs
        if (epoch + 1) % eval_iter == 0:
        
            # evaluate_metrics_epoch: CER
            cer_score_dict = evaluate_func( model, 
                                            accelerator, 
                                            valid_loader, 
                                            tokenizer
                                            )
            accelerator.print(f"Eval_Metric Epoch: {epoch} Finished")
            # evaluate_epoch_metric
            cer_score_dict["epoch"] = epoch
            accelerator.log(cer_score_dict)
            # accelerator.log(valid_metrics_epoch, step = step)
            accelerator.print(f"Eval_Metric Epoch: {epoch} Logging Finished")
            
            
        # Monitoring
        if (epoch + 1) % print_iter == 0:
            accelerator.print(f"{b_}Epoch: {epoch} | TL:{train_epoch_loss:.3e} | VL:{valid_epoch_loss:.3e} | Lowest Loss:{lowest_loss:.3e} {sr_}")
            if cer_score_dict is not None:
                accelerator.print(f"{g_}Epoch: {epoch} | CER:{cer_score_dict['cer']:.3f} {sr_}")
            print()
            
            
        # Save Best models
        if valid_epoch_loss < lowest_loss:
            accelerator.print(f"BEST Model Saving: {epoch} ")
            lowest_loss = valid_loss
            lowest_epoch = epoch
            accelerator_save_model( accelerator,
                                    model,
                                    optimizer,
                                    scheduler,
                                    epoch,
                                    step,
                                    try_name = hps.train.try_name,
                                    mode = "best",
                                    base_save_path = hps.train.model_save_base_path 
                                    )
            
        else:
            
            if hps.train.early_stop > 0 and lowest_epoch + hps.train.early_stop < epoch +1:
                accelerator.print("There is no improvement. model is shoveling. In Korean: 삽 질 중.")
                accelerator.print("Running is gonna be stopped.")
                break

    accelerator.print("The Best Validation Loss=%.3e at %d Epoch" % (lowest_loss, lowest_epoch))
    print()
    
    accelerator.end_training()
    print("This is the end")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
# ========================================================================================================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config", 
        type=str, 
        # required=True, 
        default = "/home/heiscold/korean_whisper_finetune/torch/configs/config_torch.json",
        help="path to config"
    )
    args = parser.parse_args()
    main(args)
