import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import lightning as L
from datasets import load_dataset, DatasetDict

from encoder import get_encoder


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def transforms(examples, transform):
    examples["pixel_values"] = [transform(image) for image in examples["image"]]
    return {
        "pixel_values": examples["pixel_values"],
        "labels": examples["label"],
    }

# How exactly is text data dealt with here?
# tokenized here witn the tokenizer
# in addition to the mask, details to help 
class TextData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = get_encoder()
        self.pad_token = "<PAD>"
        self.sample_texts = ["pizza with fox is the best"] #, "I came, I saw and I conquered", "Santa Claus is not real kids"]


    def train_dataloader(self):
        results = []
        for seq in self.sample_texts:
            results.append(self.tokenizer.encode(seq))
        #max_len = max([len(i) for i in results])
        #results2 = []

        #def collate_fn(batch):
        #    end_ = {"mask": [], "seq": []}
        #    for item in batch:
        #        for key_ in ["mask", "seq"]:
        #            end_[key_].append(item[key_])
        #    return end_
        #        

        #for i in results:
        #    new_ = torch.zeros(max_len)
        #    new_[:len(i)] = 1
        #    results2.append({"mask": new_, "seq": i + (max_len - len(i))*[self.pad_token]})

        def collate_fn(batch):
            return batch

        return DataLoader(results, batch_size=1, num_workers=1, pin_memory=True, drop_last=True, shuffle=True, collate_fn=collate_fn)


class ImageDataModule(L.LightningDataModule):
    data_name = None
    image_size = None
    dataset_kwargs = {}
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5] 

    def __init__(
        self,
        batch_size: int = 64,
        eval_batch_size: int | None = None,
        num_workers: int | None = None,
        add_rand_aug: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.collate_fn = None
        train_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.RandomResizedCrop(self.image_size),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transformers = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ]
        post_transforms = [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ]
        self.train_transforms = T.Compose(train_transforms + post_transforms)
        self.val_transforms = T.Compose(val_transformers + post_transforms)
        self.datasets = {}
        self.train_key = "train"
        self.val_key = "validation"
        self.test_key = "test"

    def prepare_data(self):
        load_dataset(self.data_name, num_proc=os.cpu_count() // 2)

    def setup_dataset(self, data: DatasetDict):
        return data

    def setup(self, stage=None):
        data = load_dataset(self.data_name)
        data = self.setup_dataset(data)
        train_data = data[self.train_key].with_transform(
            partial(transforms, transform=self.train_transforms)
        )
        self.datasets["train"] = train_data
        if self.test_key is not None:
            val_data = data[self.test_key].with_transform(
                partial(transforms, transform=self.val_transforms)
            )
            self.datasets["val"] = val_data
            self.val_dataloader = self._test_dataloader
        if self.test_key is not None:
            test_data = data[self.test_key].with_transform(
                partial(transforms, transform=self.val_transforms)
            )
            self.datasets["test"] = test_data
            self.test_dataloader = self._test_dataloader

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )


class CIFAR100DataModule(ImageDataModule):
    data_name = "uoft-cs/cifar100"
    image_size = 32
    num_labels = 100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        add_rand_aug = kwargs.get("add_rand_aug", False)
        self.val_key = None
        self.test_key = "test"
        # ref: https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT/blob/main/data_loader.py#L62
        train_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize([self.image_size, self.image_size]),
            T.RandomCrop(self.image_size, padding=4),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize([self.image_size, self.image_size]),
            T.CenterCrop(self.image_size),
        ]
        post_transforms = [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ]
        self.train_transforms = T.Compose(train_transforms + post_transforms)
        self.val_transforms = T.Compose(val_transforms + post_transforms)

    def setup_dataset(self, data: DatasetDict):
        data = data.remove_columns(["coarse_label"])
        data = data.rename_columns({"img": "image", "fine_label": "label"})
        return data


class TinyImageNetDataModule(ImageDataModule):
    data_name = "zh-plus/tiny-imagenet"
    image_size = 64
    num_labels = 200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        add_rand_aug = kwargs.get("add_rand_aug", False)
        self.val_key = "valid"
        self.test_key = "valid"
        train_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.RandomResizedCrop(self.image_size),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ]
        post_transforms = [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ]
        self.train_transforms = T.Compose(train_transforms + post_transforms)
        self.val_transforms = T.Compose(val_transforms + post_transforms)


def load_data(args):
    data_kwargs = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "add_rand_aug": args.add_rand_aug,
    }
    if args.data_name == "cifar100":
        return CIFAR100DataModule(**data_kwargs)
    elif args.data_name == "tiny-imagenet":
        return TinyImageNetDataModule(**data_kwargs)
    else:
        raise ValueError(f"Invalid data name: {args.data_name}")

if __name__ == '__main__':
   a = TextData().train_dataloader()
   print("hello")
   for i in a:
       print(i)
