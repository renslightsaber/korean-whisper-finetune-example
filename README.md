# korean-whisper-finetune-example
This repo is created based on [[NLP] OpenAI Whisper Fine-tuning for Korean ASR with HuggingFace Transformers](https://velog.io/@mino0121/NLP-OpenAI-Whisper-Fine-tuning-for-Korean-ASR-with-HuggingFace-Transformers) Blog Post and [colab code](https://colab.research.google.com/drive/1wSp66cLd0C6WzR9hCdvlHfIEjcd2ZfEj?usp=sharing). Actually, purpose of this repo is share how to finetune `openai/whisper-` model. I just modified the code in python script format for training in CLI environment. Also, I just added **[`wandb`](https://kr.wandb.ai/)** logging.

Click 👉 [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/Korean-Whisper-Fine-Tune-Example)


## Dataset: [**저음질 전화 음성인식 데이터**](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=571)
  - `Language`: Korean 🇰🇷
  - `sample_rate`: 8kHz

You should download this dataset and move to the `data` folder.     

### In this example code, I just used only a portion of this dataset was used for training.
- You can download the portion of this dataset at [here](https://drive.google.com/drive/folders/1eshMZ1j9H20aS6_1q3KOYgDKhd2rg_oM?usp=drive_link).
- :astonished: **data_preprocessing.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13cx7RrbsokFXe8dZ6ox8Qkel4vWzzzUF?usp=sharing): You can create pandas DataFrame with this code. (source: [colab code](https://colab.research.google.com/drive/1wSp66cLd0C6WzR9hCdvlHfIEjcd2ZfEj?usp=sharing)) And this code is included in this repo. [here](https://github.com/renslightsaber/korean-whisper-finetune-example/blob/main/data/notebooks/data_preprocessing.ipynb)! 
- :dizzy_face: **korean_whisper_fine-tuning.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PYhfStlMWrlhfF-tYJchaiJxXgwf8n39?usp=sharing): You can fine-tune whisper model at this notebook. (source: [colab code](https://colab.research.google.com/drive/1wSp66cLd0C6WzR9hCdvlHfIEjcd2ZfEj?usp=sharing))   

## Now you got ready to `train`!

### [🤗 huggingface Trainer] Train(=Fine-Tune)
First, you should move to the `hf`.  
```
cd ./hf
```

And log-in wandb with your token key in CLI. 
```
wandb login --relogin '<your-wandb-api-token>'
```

Let's train! / `BS: 16` / `LR: 5e-5` / `NVIDIA GeForce RTX 4090 (x1)`
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## [🤗 accelerate] Train(=Fine-Tune)
: you can train your code with 🤗 huggingface's [`accelerate`](https://huggingface.co/docs/accelerate/index). This package can make you feel more comfortable to use multi-gpu training, mixed precisions and others. I am sure. Of course, this code is based on 🔥 Pytorch. Also, you can use various fine-tune methods including `LoRA` fine-tune method with 🤗 huggingface's [`peft`](https://huggingface.co/docs/peft/index)

First, you should move to the `hf`.  
```
cd ./torch
```

And log-in wandb with your token key in CLI. 
```
wandb login --relogin '<your-wandb-api-token>'
```

accelerate config is needed right before training.    
: you can decide how many gpus you use and whether to use `mixed_precision` or not.
```
accelerate config
```

Let's train! / `BS: 4` / `LR: 2e-5` / `NVIDIA GeForce RTX 4090 (x1)`
: you can run training `train.py` with `accelerate launch` in CLI environment.
```
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py
```

## References
- [[NLP] OpenAI Whisper Fine-tuning for Korean ASR with HuggingFace Transformers](https://velog.io/@mino0121/NLP-OpenAI-Whisper-Fine-tuning-for-Korean-ASR-with-HuggingFace-Transformers)
- [colab code](https://colab.research.google.com/drive/1wSp66cLd0C6WzR9hCdvlHfIEjcd2ZfEj?usp=sharing)
- 🤗 [huggingface Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#api-reference%20][%20transformers.Trainer)
- 🤗 [huggingface TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- 🤗 [Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/v4.24.0/perf_train_gpu_one)
- 🤗 [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one)
