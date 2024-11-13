import torch
from transformers import WhisperForConditionalGeneration

def get_whisper_model(model_name, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # model_name: "openai/whisper-base", ...

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    if device is not None:
        return model.to(device)
    else:
        return model