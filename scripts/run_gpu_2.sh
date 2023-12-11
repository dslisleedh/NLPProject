CUDA_VISIBLE_DEVICES=1 python train.py model.name=RN50x16\
    model.m2_loss_weight=0.1\
    optimizer.lr=5e-5\
    scheduler.eta_min=5e-6

CUDA_VISIBLE_DEVICES=1 python train.py model.name=RN50x16\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    model.n_prompt=1\
    model.logit_scale=4.6052\
    train.batch_size=256