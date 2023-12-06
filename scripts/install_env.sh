# Remove Temp file
yes | conda clean --all

# Since BLIP and MMdet has conflict on opencv-python, we should split each env
yes | conda create -n nlpproject_mmdet python=3.10
yes | conda create -n nlpproject_blip python=3.10

# MMdet
conda activate nlpproject_mmdet
yes | conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

yes | pip install -U openmim
yes | mim install mmengine
yes | mim install "mmcv>=2.0.0"
yes | mim install mmdet
yes | pip install ftfy regex tqdm

# BLIP and CLIP
conda activate nlpproject_blip
yes | conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

yes | pip install transformers
yes | pip intsall hydra-core
yes | pip intsall hydra-colorlog
yes | pip install torchmetrics
yes | pip install timm
yes | pip install tb-nightly
yes | pip install einops
yes | pip install ftfy

git clone https://github.com/openai/CLIP.git
