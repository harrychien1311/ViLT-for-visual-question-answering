from vilt.datasets import VQADataset
from .datamodule_base import BaseDataModule
from collections import defaultdict
from torch.utils.data import DataLoader

class VQADataModule(BaseDataModule):
    def __init__(self, _config):
        super().__init__(_config)
        self.data_root = _config["data_root"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.train_transform_keys = _config["train_transform_keys"]
        self.test_transform_keys = _config["val_transform_keys"]
        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

    @property
    def dataset_cls(self):
        return VQADataset

    @property
    def dataset_name(self):
        return "vqa"

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            collate = self.mlm_collator,
            data_root = self.data_root,
            transform_keys = self.train_transform_keys,
            image_size=self.image_size,
            split="train",
            max_text_len=self.max_text_len,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            collate = self.mlm_collator,
            data_root = self.data_root,
            transform_keys = self.test_transform_keys,
            image_size=self.image_size,
            split="val",
            max_text_len=self.max_text_len,
        )
    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            collate = self.mlm_collator,
            data_root = self.data_root,
            transform_keys = self.test_transform_keys,
            image_size=self.image_size,
            split="test",
            max_text_len=self.max_text_len,
        )
    
    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate
        )
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate
        )
        return loader
