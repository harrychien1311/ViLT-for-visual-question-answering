import torch
from .base_dataset import BaseDataset
import json
from PIL import Image
from vilt.transforms import _transforms
import os
from torchvision import transforms
from vilt.transforms.utils import inception_normalize

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, collate, data_root: str, transform_keys: list, image_size: int, split="", max_text_len=40):
        assert split in ["train", "val", "test"]
        super().__init__()
        self.split = split
        self.data_root = os.path.join(data_root, self.split)
        self.datapath = f"{self.split}_encode_annotation.json"
        with open(self.datapath, "r") as f:
            self.data = json.load(f)
        self.image_size = image_size
        self.transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            inception_normalize,
        ]
        )
        self.max_text_len = max_text_len
        self.mlm_collator=collate
    def get_image(self, index):
        img_path = os.path.join(self.data_root, self.data["image_id"][index])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        img_size = img.size()
        if img_size[1] != self.image_size or img_size[2] != self.image_size:
            print(f"Image size: {img.size()}")
        return img
    
    def get_question(self, index):
        question = self.data["question"][index]
        encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return encoding
    
    def get_answer(self, index):
        return self.data["answer"][index]
    
    def __len__(self):
        return len(self.data["image_id"])
    
    def __getitem__(self, index):
        image_tensor = self.get_image(index)
        question = self.get_question(index)

        if self.split != "test":
            answer = self.get_answer(index)
        else:
            answer = ""
        return {
            "image": image_tensor,
            "question": question,
            "vqa_answer": answer,
        }
    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # print(type(dict_batch["image"]))
        # dict_batch["image"] = torch.tensor(dict_batch["image"])
        #print(dict_batch["question"])

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()
        #print(dict_batch["image"])
        img = dict_batch["image"]
        img_sizes += [i.shape for i in img if i is not None]
        #print(img_sizes)

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]

            new_images = torch.zeros(batch_size, 3, max_height, max_width)

            for bi in range(batch_size):
                orig = img[bi]
                new_images[bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        new_answers = torch.zeros(batch_size, 1)
        for bi in range(batch_size):
            orig_answer  = dict_batch["vqa_answer"][bi]
            new_answers[bi,:] = torch.tensor(orig_answer)
        dict_batch["vqa_answer"] = new_answers

        txt_keys = [k for k in list(dict_batch.keys()) if "question" in k]
        if len(txt_keys) != 0:
            encodings = [[d for d in dict_batch[txt_key]] for txt_key in txt_keys]
            #print(encodings)
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                encodings = [d for d in dict_batch[txt_key]]

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                # print("type of text_ids: {}", type(dict_batch[f"{txt_key}_ids"]))
                # print("type of text_labels: {}", type(dict_batch[f"{txt_key}_labels"]))
                dict_batch.pop("question")
                
        return dict_batch
