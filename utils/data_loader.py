from torchvision import datasets, transforms
import os
import torch
import logging

def set_logging(name=None):
    rank = int(os.getenv("RANK", -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)

class DataLoader:
    def __init__(self, fold_dir, batch_size, fold):
        self.fold_dir = fold_dir
        self.batch_size = batch_size
        self.fold = fold
    
    def train_loader(self):
        train_path = os.path.join(self.fold_dir, "Fold" + str(self.fold), "Train")
        train_data = datasets.ImageFolder(train_path,       
                            transform=self.data_transforms("train"))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                                shuffle = True, num_workers = 0)
        return train_loader

    def val_loader(self):
        val_path = os.path.join(self.fold_dir, "Fold" + str(self.fold),  "Val")
        val_data = datasets.ImageFolder(val_path,
                            transform = self.data_transforms("val"))
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = self.batch_size,
                                            shuffle = False, num_workers = 0)
        return val_loader


    def test_loader(self):
        test_path = os.path.join(self.fold_dir, "Fold" + str(self.fold),  "Test")
        test_data = datasets.ImageFolder(test_path,
                            transform = self.data_transforms("test"))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = self.batch_size,
                                            shuffle = False, num_workers = 0)
        return test_loader

    def data_transforms(self, stage):
        mean_nums = [0.485, 0.456, 0.406]
        std_nums = [0.229, 0.224, 0.225]
        image_size = 224

        data_transforms = {"train":transforms.Compose([
                                    transforms.Resize((image_size, image_size)), #Resizes all images into same dimension
                                    transforms.ToTensor(), # Coverts into Tensors
                                    transforms.Normalize(mean = mean_nums, std=std_nums)]), # Normalizes
                            "val": transforms.Compose([
                                    transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean_nums, std = std_nums)
                            ]),
                            "test": transforms.Compose([
                                    transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean_nums, std = std_nums)
                            ])
                            }
        return data_transforms[stage]
        
