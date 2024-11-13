import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio import transforms

from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Dataset
class SimpleASRDataset(Dataset):
    def __init__(self,
                    df,
                    processor,
                    resample_rate=16000
                    ):
        self.df = df
        self.processor = processor
        self.resample_rate = resample_rate
        self.audios = self.df.raw_data.to_list()
        self.transcripts = self.df.transcript.to_list()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        # audio
        wav, sample_rate = torchaudio.load(self.audios[idx])
        transform = transforms.Resample(sample_rate, self.resample_rate)
        resampled_waveform = transform(wav)

        input_features = self.processor(resampled_waveform.squeeze(0),     # Batch Size 제거
                                        sampling_rate= self.resample_rate, # whisper-small은 sampling rate을 16000인 데이터에 대해서만 input으로 받는다고 합니다.
                                        return_tensors="pt"
                                        ).input_features.squeeze(0)

        # transcripts
        labels = self.processor.tokenizer(  self.transcripts[idx],
                                            padding=True,
                                            truncation=True,
                                            return_tensors="pt").input_ids.squeeze(0)

        # return
        return {'input_features': input_features, 'labels': labels}
    
    
## Collate_function
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
        # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenize된 레이블 시퀀스를 가져온다.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
        # 해당 토큰은 이후 언제든 추가할 수 있다.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
## DataLoaders
def prepare_loaders(df,
                    train_ratio,
                    processor,
                    batch_size = (16, 8),
                    ):

    total_rows = df.shape[0]
    cutoff = int(train_ratio * total_rows)
    print(cutoff)

    # DataFrame Indexing
    train_df = df[:cutoff].reset_index(drop = True)
    valid_df = df[cutoff:].reset_index(drop = True)

    # MyDataset
    train_ds = SimpleASRDataset(df = train_df, processor = processor)
    valid_ds = SimpleASRDataset(df = valid_df, processor = processor)

    # DataLoader
    train_dataloader = DataLoader(train_ds, batch_size = batch_size[0], shuffle = True, collate_fn = DataCollatorSpeechSeq2SeqWithPadding(processor =processor))
    valid_dataloader = DataLoader(valid_ds, batch_size = batch_size[1], shuffle = False, collate_fn = DataCollatorSpeechSeq2SeqWithPadding(processor =processor))
    
    print("Dataset Completed")
    return train_dataloader, valid_dataloader


