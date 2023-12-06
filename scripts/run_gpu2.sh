# Run Fine-tuning With Mixup

CUDA_VISIBLE_DEVICES=2 python train.py model.name=RN50\
    optimizer.lr=0.00005\
    scheduler.eta_min=0.000001\
    model.m2_loss_weight=0.1

CUDA_VISIBLE_DEVICES=2 python train.py model.name=RN101\
    optimizer.lr=0.00005\
    scheduler.eta_min=0.000001\
    model.m2_loss_weight=0.1

CUDA_VISIBLE_DEVICES=2 python train.py model.name=RN50x4\
    optimizer.lr=0.00005\
    scheduler.eta_min=0.000001\
    model.m2_loss_weight=0.1

CUDA_VISIBLE_DEVICES=2 python train.py model.name=RN50x16\
    optimizer.lr=0.00005\
    scheduler.eta_min=0.000001\
    model.m2_loss_weight=0.1

CUDA_VISIBLE_DEVICES=2 python train.py model.name=RN50x64\
    optimizer.lr=0.00005\
    scheduler.eta_min=0.000001\
    model.m2_loss_weight=0.1