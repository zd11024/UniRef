import math
import torch
from PIL import Image
import ruamel.yaml as yaml
from transformers import AutoTokenizer
from torchvision import transforms
from models.uniref import UniRef, UniRefDecoder

device = torch.device('cuda')
class rec_inference:

    def __init__(self, checkpoint, image_res=384, patch_size=16):
        self.image_res = image_res
        self.patch_size = patch_size
        self.num_patch = self.image_res // self.patch_size

        # setup
        config = yaml.load(open('configs/uniref_finetune.yaml', 'r'), Loader=yaml.Loader)
        # config['text_encoder'] = 'bert-base-uncased'

        # transform
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_encoder'])

        # model
        model = UniRef(config=config,load_vision_params=False)
        model.load_pretrained(checkpoint, config, is_eval=True)
        model = model.to(device)
        self.model = model
        

    def inference(self, img_path, text):
        """
        Input: image path, text
        Retuen: [x, y, w, h]
        """
        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        image = self.transform(image)
        text_input = self.tokenizer(text, padding='longest', return_tensors="pt")

        image = image.unsqueeze(0).to(device)
        input_ids = text_input['input_ids'].to(device)
        attention_mask = text_input['attention_mask'].to(device)
        
        with torch.no_grad():
            pred = self.model.predict(image, input_ids, attention_mask)

        coord = pred[0]
        coord[0::2] *= W
        coord[1::2] *= H
        coord[0] -= coord[2] / 2
        coord[1] -= coord[3] / 2
        
        return coord

pipeline = rec_inference('checkpoints/reg_refcoco+.th')
sent = pipeline.inference('images/coco/train2017/000000581563.jpg', 'traffic light')
print(sent)
