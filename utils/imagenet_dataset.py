import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


class ImageNetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
    ):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.LANCZOS),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.samples = []
        self.targets = []
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.endswith(".JPEG"):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append(img_path)
                    self.targets.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, target


def get_imagenet_dataloader(
    rank: int,
    world_size: int,
    data_dir: str = "data/",
    train_batch_size: int = 12,
    val_batch_size: int = 12,
    num_workers: int = 4,
    shuffle: bool = True,
):
    train_ds = ImageNetDataset(root_dir=data_dir, split="train")
    val_ds = ImageNetDataset(root_dir=data_dir, split="validation")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=shuffle)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    ds = ImageNetDataset(root_dir="data/", split="train")
    print(len(ds))
    print(ds[0])
