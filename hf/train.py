import os
import gc
import wandb
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchaudio

# Huggingface: Model Download
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

from hf_dataset import *

# whisper model
from model import *

# others
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


def main(args):
    
    # HParams
    config_path = args.config # "/home/heiscold/tse_vc_supervised/configs/config.json"
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hps = HParams(**config)
    print(f"{b_} config loaded {sr_}")
    print()
    
    model_name = hps.train.model_name
    lang = hps.train.lang
    print(f"{r_} model_name: {model_name}, lang: {lang}, task: {hps.train.task} {sr_}")
    print()
    # model_name = "openai/whisper-small"
    # lang = "Korean"

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
    
    # Dataset
    train_ds, valid_ds = prepare_datasets(df= df, train_ratio = hps.train.train_ratio, processor = processor)
    print(f"{m_} Dataset for hf Trainer loaded {sr_}")
    
    # Model Downloaded
    model = get_whisper_model(  model_name = model_name,
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                )
    
    # MOdel Downloaded
    print(f"{b_} Model Downloaded {sr_}")
    
    # model cuda? Check!
    print(f"{y_} Model is on GPU?: {next(model.parameters()).is_cuda} {sr_}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr =  hps.train.lr)
    print(f"{r_} Optimizer Defined {sr_}")
    
    # Scheduler
    steps_per_epoch = len(train_ds) // hps.train.train_batch_size
    print(f"{s_} steps_per_epoch: {steps_per_epoch} {sr_}")
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = hps.train.lr,
                                                    steps_per_epoch = int(steps_per_epoch),
                                                    epochs = hps.train.n_epochs
                                                    )
    print(f"{s_} Scheduler Defined {sr_}")
    
    # evaluate metric function
    def compute_metrics(pred):
        # import evaluate
        metric = evaluate.load('cer')

        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # pad_token을 -100으로 치환
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # metrics 계산 시 special token들을 빼고 계산하도록 설정
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}
    
    print(f"{r_} Evaluation Metric Defined {sr_}")
    
    # wandb Login at CLI
    # !wandb login --relogin '<개인 API 키>'
    
    # wandb init
    run = wandb.init(project= hps.train.project_name, # 'Korean-Whisper-Fine-Tune-Example',
                    config = config,
                    job_type = 'Training',
                    name = hps.train.try_name,
                    anonymous = 'must'
                    )
    
    
    # Huggingface TrainingArguments
    max_iters = int(steps_per_epoch * hps.train.n_epochs)
    print(f"{y_} Max iters = {max_iters} {sr_}")

    # from transformers import Seq2SeqTrainingArguments

    training_args = Seq2SeqTrainingArguments(
        output_dir = hps.train.model_save_path, # 원하는 경로
        per_gpu_train_batch_size = hps.train.train_batch_size,
        gradient_accumulation_steps = 1,
        # learning_rate=1e-5,                   # Optimizer를 이전 셀에서 선언해서 입력하지 않습니다.
        # warmup_steps=500,                     # Scheduler를 이전 셀에서 선언해서 입력하지 않습니다.
        # evaluation_strategy = 'epoch',        # epoch 기준으로 평가할 수 있습니다.
        evaluation_strategy = 'steps',
        eval_steps = 100,
        # num_train_epochs= config.n_epochs,
        max_steps = max_iters,                  # epoch 대신 설정
        seed = hps.train.seed,                  # 이전에서 선언했지만, 여기서 한 번 더 해도 상관없습니다.
        gradient_checkpointing=True,
        group_by_length = True,
        # fp16=True,                              # mixed_precision="fp16"
        bf16 = True,                            # mixed_precision="bf16"
        per_gpu_eval_batch_size= hps.train.valid_batch_size,
        # per_device_eval_batch_size= hps.train.valid_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        logging_strategy="steps",
        logging_steps=5,
        report_to=["wandb"],
        load_best_model_at_end = True,
        metric_for_best_model = "cer",            # 한국어의 경우 'wer'보다는 'cer'이 더 적합할 것
        greater_is_better = False,
        save_strategy = "steps",                  # 'epoch': epoch 기준으로 저장 가능
        save_total_limit = hps.train.save_total_limit # 3,
        # push_to_hub=False,
    )
    print(f"{b_} Huggingface TrainingArguments Completed {sr_}")


    # Huggingface Trainer
    # from transformers import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset = train_ds,
        eval_dataset = valid_ds,  # or "test"
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics = compute_metrics,
        tokenizer = processor.feature_extractor,
        optimizers = (optimizer, scheduler),
    )
    print(f"{b_} Huggingface Trainer Completed {sr_}")
    print()

    # Training Start!
    print(f"{y_} Training Start {sr_}")
    trainer.train()
    print(f"{s_} Training Finished {sr_}")
    print()
    
    # Evaluate
    trainer.evaluate()
    print(f"{m_} Evaluation Finished {sr_}")
    print()
    
    # Model Save
    trainer.model.save_pretrained(hps.train.model_save_path)
    tokenizer.save_pretrained(hps.train.model_save_path)
    
    print("Train Completed")
    
    # wandb finished
    run.finish()
    
    # Training Finished
    torch.cuda.empty_cache()
    _ = gc.collect()


# ========================================================================================================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config", 
        type=str, 
        # required=True, 
        default = "/home/heiscold/korean_whisper_finetune/hf/configs/config.json",
        help="path to config"
    )
    args = parser.parse_args()
    main(args)