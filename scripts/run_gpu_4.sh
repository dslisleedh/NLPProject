CUDA_VISIBLE_DEVICES=3 python train.py model.name=RN50x16\
    model.prompt_learning=True\
    model.prompt_from_visual_tokens=True\
    model.n_prompt=1\
    train.batch_size=256
    