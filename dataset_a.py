import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
# from albumentations.pytorch import ToTensorV2

from config import GlobalConfig

def get_train_transform(img_size):
    transform = A.Compose([

        A.Resize(img_size, img_size),

        # A.OneOf([
        #     A.Resize(img_size, img_size),
        #     A.RandomCrop(width=img_size, height=img_size),
        # ], p=1),
        
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),

        # A.Rotate(180, p=0.8),
        
        # A.Blur(p=1),
        # A.CLAHE(p=0.5),
        # A.ColorJitter(brightness=0.2, contrast=0.9, saturation=0.2, p=1),

        # A.ShiftScaleRotate(p=1),

        # A.OneOf([
        #     A.OpticalDistortion(distort_limit=1.0),
        #     A.GridDistortion(num_steps=5, distort_limit=1.),
        #     A.ElasticTransform(alpha=3),
        # ], p=1),

        # A.OneOf([
        #     A.CLAHE(p=0.5),
        #     A.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, p=0.5),
        # ], p=1),
    
        # A.CoarseDropout(max_holes=3, max_height=32, max_width=32, p=1),
        # A.CoarseDropout(max_holes=10, max_height=32, max_width=32, p=0.7),

        # A.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225]),
        # A.ToTensorV2()
    ])
    return transform

def get_valid_transform(img_size):
    transform = A.Compose([
        A.Resize(img_size, img_size),
    ])
    return transform

def get_normalize_transform():
    transform = A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    return transform


class ImageDataset(Dataset):
    def __init__(self, df, path, transform=None):
        print('Init dataset A')
        self.df = df
        self.path = path
        self.transform = transform
        self.normalize = get_normalize_transform()

        self.image_ids = df.StudyInstanceUID
        self.labels = df[GlobalConfig.target_columns]     
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        image_id = self.image_ids.iloc[index]
        label = self.labels.iloc[index].values.astype(float)
        
        pillow_image = Image.open(self.path+image_id+'.jpg')
        image = np.array(pillow_image)
        
        if self.transform:
            image = self.transform(image=image)['image']

        image = image.astype(np.float32)
        # image /= 255

        # for frayscale images
        # image = np.repeat(image[np.newaxis,:], 3, axis=0)
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
   
        image = self.normalize(image=image)['image']

        image = image.transpose(2, 0, 1)
            
        return image, label

    def get_image_to_show(self, image):
        return image[0]
        # return image.transpose(1, 2, 0)


class TestImageDataset(Dataset):
    def __init__(self, df, path, transform=None):
        print('Init test dataset A')
        self.df = df
        self.path = path
        self.transform = transform
        
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row.image_id
     
        pillow_image = Image.open(self.path+image_id)
        image = np.array(pillow_image)
        
        if self.transform:
            image = self.transform(image=image)['image']

        image = image.astype(np.float32)
        image /= 255
        image = image.transpose(2, 0, 1)
            
        return image

    def get_image_to_show(self, image):
        return image.transpose(1, 2, 0)