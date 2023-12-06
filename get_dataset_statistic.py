from collections import Counter

from omegaconf import OmegaConf


if __name__ == '__main__':
    metafile = OmegaConf.load('./hpitp_dataset/metafile.yaml')
    
    print('LLM_Question: ', metafile['question'])
    question = metafile['question']
    
    samples = metafile['train_samples'] + metafile['test_samples']
    n_samples = len(samples)
    print("Number of samples: ", n_samples)
    img_dir = [sample['img_dir'] for sample in samples]
    text_pair = [sample['answer'] for sample in samples]
    
    datasets = ['mpii', 'DF2K', 'LSDIR', 'ImageNet']
    
    for dataset in datasets:
        n_d_sampels = len([img for img in img_dir if dataset in img])
        print(f'Number of {dataset} samples: ', n_d_sampels)
        
    text_pair = dict(Counter(text_pair))
    # Sort by value
    text_pair = {k: v for k, v in sorted(text_pair.items(), key=lambda item: item[1], reverse=True)}
    step = 0
    for text, n in text_pair.items():
        print(f'{text}: {n}')
        step += 1
        if step > 100:
            break
