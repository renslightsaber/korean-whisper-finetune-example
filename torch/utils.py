import os
import gc
import json
import torch

# ========================================================================================================================================================== #

def accelerator_save_model( accelerator,
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            step,
                            try_name,
                            mode = "best",
                            base_save_path ="/home/heiscold/korean_whisper_finetune/torch/ckpt/"
                            ):
    
    # os.mkdir 없으면 경로 생성
    os.makedirs(f"{base_save_path}{try_name}/{mode}/model/", exist_ok=True)
    os.makedirs(f"{base_save_path}{try_name}/{mode}/optimizer/", exist_ok=True)
    os.makedirs(f"{base_save_path}{try_name}/{mode}/scheduler/", exist_ok=True)
    
    model_save_path = f"{base_save_path}{try_name}/{mode}/model/"
    optimizer_save_path = f"{base_save_path}{try_name}/{mode}/optimizer/"
    scheduler_save_path = f"{base_save_path}{try_name}/{mode}/scheduler/"
    
    
    if accelerator.is_main_process:
        
        # 모든 GPU 연산을 동기화
        # torch.cuda.synchronize()
        # accelerator.wait_for_everyone()
        
        # Unwrap: model, optimizer, scheduler.
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_optimizer = accelerator.unwrap_model(optimizer) 
        unwrapped_scheduler = accelerator.unwrap_model(scheduler) 
        
        # Save model, optimizer, scheduler.
        # accelerator.save(unwrapped_model.state_dict() , model_save_path + f"model_{epoch:03d}_{step:03d}.pth")
        unwrapped_model.save_pretrained(model_save_path + f"model_{epoch:03d}_{step:03d}/")
        
        accelerator.save(unwrapped_optimizer.state_dict(), optimizer_save_path  + f"optimizer_{epoch:03d}_{step:03d}.pth")
        accelerator.save(unwrapped_scheduler.state_dict(), scheduler_save_path + f"scheduler_{epoch:03d}_{step:03d}.pth")
        accelerator.print(f"{mode}'s model and others @ step:{step:03d} of epoch: {epoch:03d} are saved!")
        print()
        
    else:
        # accelerator.wait_for_everyone()
        accelerator.print(f"{mode}'s model and others are saved? Not Sure @ step:{step:03d} of epoch: {epoch:03d} are saved!")
        print()    

# ========================================================================================================================================================== #

# 임계값 설정 (예: GPU 메모리의 90% 이상 사용 시)
def condition_for_memory_cleanup(threshold=0.96):
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    return allocated_memory / total_memory > threshold

# ========================================================================================================================================================== #

class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

# ========================================================================================================================================================== #
