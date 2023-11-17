import sys
sys.path.append('./src/')
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from albumentations import Compose
from utils.dataset import SeedDataset

sys.path.append('./src/')
from config import train_dir, val_dir, annotation_path
from utils.transforms import collate_fn, train_transforms, val_transforms

class FRCNNDataModule(LightningDataModule):
    """LightningDataModule used for training EffDet
     This supports COCO dataset input
    Args:
        img_dir: image directory
        annotation_dir (Path): annoation directory
        num_workers (int): number of workers to use for loading data
        batch_size (int): batch size
        img_size (int): image size to resize input data to during data
         augmentation
    """

    def __init__(
            self, 
            train_dir: str, 
            val_dir: str, 
            annotation_path: str, 
            train_transforms: Compose=None, 
            val_transforms:Compose=None, 
            train_subset: list=None, 
            val_subset: list=None, 
            num_workers: int=2, 
            batch_size: int=1, 
            img_size: int=None, 
            collate_fn=collate_fn
            ):
        super().__init__()
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.annotation_path = annotation_path
        self.collate_fn = collate_fn

    def train_dataset(self) -> SeedDataset:
        return SeedDataset(
            image_dir=self.train_dir, 
            annotation_path=self.annotation_path, 
            resize_dims=self.img_size, 
            transforms=self.train_transforms, 
            subset=self.train_subset
            )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> SeedDataset:
        return SeedDataset(
            image_dir=self.val_dir, 
            annotation_path=self.annotation_path, 
            resize_dims=self.img_size, 
            transforms=self.val_transforms, 
            subset=self.val_subset
            )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return val_loader

    # @staticmethod
    # def collate_fn(batch):
    #     batch = [x for x in batch if x is not None]
    #     images, targets = tuple(zip(*batch))
    #     images = torch.stack(images)
    #     images = images.float()

    #     boxes = [target["bboxes"].float() for target in targets]
    #     labels = [target["labels"].float() for target in targets]
    #     img_size = torch.tensor([target["img_size"] for target in targets]).float()
    #     img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

    #     annotations = {
    #         "bbox": boxes,
    #         "cls": labels,
    #         "img_size": img_size,
    #         "img_scale": img_scale,
    #     }

    #     return images, annotations, targets


if __name__ == '__main__':
    mod1 = FasterRCNNModule(train_dir, val_dir, annotation_path, train_transforms=train_transforms, val_transforms=val_transforms)