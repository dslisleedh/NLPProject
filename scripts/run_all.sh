# 1. fine tuning all clip model
# 2. Prompt Learning
# 3. Prompt Learning with visual conditioning

python train.py model.name=openai/clip-vit-large-patch14
python train.py model.name=openai/clip-vit-base-patch16
python train.py model.name=openai/clip-vit-base-patch32
python train.py model.name=openai/clip-vit-large-patch14-336\
    dataset.img_size=336

python train.py model.name=openai/clip-vit-large-patch14-336\
    dataset.img_size=336\
    model.prompt_learning=True\
    train.batch_size=32
    
python train.py model.name=openai/clip-vit-large-patch14-336\
    dataset.img_size=336\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    train.batch_size=32\
    hydra.job.env_set.CUDA_VISIBLE_DEVICES=2

python train.py model.name=openai/clip-vit-large-patch14-336\
    dataset.img_size=336\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    model.n_prompt=64\
    train.batch_size=32\
    hydra.job.env_set.CUDA_VISIBLE_DEVICES=3