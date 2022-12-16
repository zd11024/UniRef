import math
import torch
from PIL import Image
import ruamel.yaml as yaml
from transformers import AutoTokenizer
from torchvision import transforms
from models.uniref import UniRef, UniRefDecoder

device = torch.device('cuda')
class reg_inference:

    def __init__(self, checkpoint, image_res=384, patch_size=16):
        self.image_res = image_res
        self.patch_size = patch_size
        self.num_patch = self.image_res // self.patch_size

        # setup
        config = yaml.load(open('configs/uniref_finetune.yaml', 'r'), Loader=yaml.Loader)
        config['text_encoder'] = 'bert-base-uncased'

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
        gen_model = UniRefDecoder.from_pretrained(config, self.tokenizer, model)
        self.model = gen_model

    def get_image_attns(self, x, y, w, h):
        x_min = min(math.floor(x / self.patch_size), self.num_patch - 1)
        x_max = max(x_min+1, min(math.ceil((x+w) / self.patch_size), self.num_patch))  # exclude

        y_min = min(math.floor(y / self.patch_size), self.num_patch - 1)
        y_max = max(y_min+1, min(math.ceil((y+h) / self.patch_size), self.num_patch))  # exclude

        image_atts = [0] * (1 + self.num_patch ** 2)
        image_atts[0] = 1  # always include [CLS]
        for j in range(x_min, x_max):
            for i in range(y_min, y_max):
                index = self.num_patch * i + j + 1
                assert (index > 0) and (index <= self.num_patch ** 2), f"patch index out of range, index: {index}"
                image_atts[index] = 1

        return image_atts

    def preprocess_input(self, img_path, bbox):

        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        image = self.transform(image)
        
        x,y,w,h = bbox
        # resize applied
        x = self.image_res / W * x
        w = self.image_res / W * w
        y = self.image_res / H * y
        h = self.image_res / H * h

        center_x = x + 1 / 2 * w
        center_y = y + 1 / 2 * h

        target_bbox = torch.tensor([center_x / self.image_res, center_y / self.image_res,
                                    w / self.image_res, h / self.image_res], dtype=torch.float)

        image_atts = torch.tensor(self.get_image_attns(x, y, w, h))

        image = image.unsqueeze(0)
        target_bbox = target_bbox.unsqueeze(0)
        image_atts = image_atts.unsqueeze(0)

        return image, target_bbox, image_atts
        

    def inference(self, img_path, bbox):
        """
        Input: image path
        bbox: [x, y, w, h], w is width, h is hidth
        """
        image, target_bbox, image_atts = self.preprocess_input(img_path, bbox)

        batch_size = image.size(0)
        image = image.to(device)
        image_atts = image_atts.to(device)
        idx_to_group_img = torch.arange(batch_size).to(device)
        input_ids = torch.zeros([batch_size, 1], dtype=torch.int).to(device) + self.tokenizer.cls_token_id
        attention_mask = torch.ones([batch_size, 1], dtype=torch.int).to(device)

        decode_args = {
            'max_length': 20,
            'bos_token_id': self.tokenizer.cls_token_id,
            'eos_token_id': self.tokenizer.sep_token_id,
            'min_length': 2,
            'num_beams': 5,
            'early_stopping': True,
        }

        with torch.no_grad():

            output_sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                image_atts=image_atts,
                idx_to_group_img=idx_to_group_img,
                use_new_segment=0,
                **decode_args
            )

        sent = self.tokenizer.decode(output_sequences[0].tolist(), skip_special_tokens=True)
        return sent

pipeline = reg_inference('checkpoints/reg_refcoco+.th')
sent = pipeline.inference('images/coco/train2017/000000581563.jpg', [100, 200, 200, 300])
print(sent)
