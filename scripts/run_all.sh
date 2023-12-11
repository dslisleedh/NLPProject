# Fine tuning and caluclate non-trained accuracy
CUDA_VISIBLE_DEVICES=0 python train.py model.name=RN50x16

# Fine tuning + M2 Loss
CUDA_VISIBLE_DEVICES=1 python train.py model.name=RN50x16\
    model.m2_loss_weight=0.1

# Prompt Learning
CUDA_VISIBLE_DEVICES=2 python train.py model.name=RN50x16\
    model.prompt_learning=True\
    model.n_prompt=1\
    train.batch_size=256

# Prompt Learning + Q-former
CUDA_VISIBLE_DEVICES=3 python train.py model.name=RN50x16\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    model.n_prompt=1\
    train.batch_size=256


# Q-Fromer ablation
# 1. Q-Former with 16 prompts
CUDA_VISIBLE_DEVICES=0 python train.py model.name=RN50x16\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    model.n_prompt=16\
    train.batch_size=256

# 2. Q-Former with logit scale 4.6052
CUDA_VISIBLE_DEVICES=1 python train.py model.name=RN50x16\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    model.n_prompt=1\
    model.logit_scale=4.6052\
    train.batch_size=256

# 3. Q-former with smaller batch size
CUDA_VISIBLE_DEVICES=2 python train.py model.name=RN50x16\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    model.n_prompt=1\
    train.batch_size=16
