import os
from utils import check_image_file
from PIL import Image
import torchvision.transforms as transforms


class Dataset(object):
    def __init__(self, images_dir, image_size, upscale_factor):
        self.filenames = [os.path.join(images_dir, x) for x in os.listdir(images_dir) if check_image_file(x)]
        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        hr = self.hr_transforms(Image.open(self.filenames[idx]).convert("RGB"))
        lr = self.lr_transforms(hr)
        return lr, hr

    def __len__(self):
        return len(self.filenames)
