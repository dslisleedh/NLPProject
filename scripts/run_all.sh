# 1. fine tuning all clip model
# 2. Prompt Learning
# 3. Prompt Learning with visual conditioning

CUDA_VISIBLE_DEVICES=0 python train.py model.name=openai/clip-vit-large-patch14
CUDA_VISIBLE_DEVICES=0 python train.py model.name=openai/clip-vit-base-patch16
CUDA_VISIBLE_DEVICES=0 python train.py model.name=openai/clip-vit-base-patch32
CUDA_VISIBLE_DEVICES=0 python train.py model.name=openai/clip-vit-large-patch14-336\
    dataset.img_size=336

CUDA_VISIBLE_DEVICES=0 python train.py model.name=openai/clip-vit-large-patch14-336\
    dataset.img_size=336\
    model.prompt_learning=True\
    train.batch_size=32
    
CUDA_VISIBLE_DEVICES=0 python train.py model.name=openai/clip-vit-large-patch14-336\
    dataset.img_size=336\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    train.batch_size=32
