import torch
from PIL import Image

# from lavis.models import load_model_and_preprocess
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import os 
from omegaconf import OmegaConf
from tqdm import tqdm

img_extension = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] # We only used png to make the dataset but just in case


def main(image_path: str):
    # question = 'Please describe the specific pose of the man in this image. Describe in one sentence and only consider pose itself not additional information.'
    question =  "Question: Describe pose in images and exclude everything not related with pose. Answer:" # You must ignore what the person is wearing, grapping or background in images."
    
    if image_path == './hpitp_dataset/images/':
        metafile_dir = './hpitp_dataset/'
    else:
        metafile_dir = image_path

    os.makedirs(metafile_dir, exist_ok=True)
    
    imgs_dir = [os.path.join(image_path, img) for img in os.listdir(image_path) if os.path.splitext(img)[1] in img_extension]
    imgs_dir.sort()
    print(f'Found {len(imgs_dir)} images')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16).to(device)
    metafile_dict = {
        "question": question,
        "samples": {}
    }
    
    for img_dir in tqdm(imgs_dir[:50]):
        img = Image.open(img_dir).convert("RGB")
        img_name = os.path.splitext(os.path.basename(img_dir))[0]
        
        # Saleforce API        
        # image = vis_processors["eval"](img).unsqueeze(0).to(device)
        # question = txt_processors["eval"](question)
        # samples = {"image": image, "text_input": question}
        # answer = model.predict_answers(samples=samples, inference_method="generate")[0]
        
        # HuggingFace API
        inputs = processor(img, text=question, return_tensors="pt").to(device, torch.float16)
        # inputs = processor(img, return_tensors="pt").to(device, torch.float16)
        
        answer_ids = model.generate(**inputs, max_new_tokens=20)
        answer = processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        print(answer)
        
        metafile_dict["samples"][img_name] = answer

    # Save metafile
    OmegaConf.save(OmegaConf.create(metafile_dict), f'{metafile_dir}/metafile.yaml')
    
    
if __name__ == '__main__':
    if os.environ.get('IMAGE_PATH') is None:
        image_path = './hpitp_dataset/images/'
    else:
        image_path = os.environ.get('IMAGE_PATH')
        
    main(image_path)