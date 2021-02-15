import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
# from albumentations.pytorch import ToTensorV2

def get_train_transform(img_size):
    transform = A.Compose([

        A.OneOf([
            A.Resize(img_size, img_size),
            A.RandomCrop(width=img_size, height=img_size),
        ], p=1),
        
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),

        # A.Rotate(180, p=0.8),
        
        # A.Blur(p=0.2),
        # A.ColorJitter(p=0.4),
    
        # A.CoarseDropout(max_holes=3, max_height=32, max_width=32, p=1),
        # A.CoarseDropout(max_holes=10, max_height=32, max_width=32, p=0.7),

        # A.ToTensorV2(),
        # A.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_valid_transform(img_size):
    transform = A.Compose([
        A.Resize(img_size, img_size),
    ])
    return transform

class ImageDataset(Dataset):
    def __init__(self, df, path, transform=None):
        print('Init dataset A')
        self.df = df
        self.path = path
        self.transform = transform
        
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row.image_id
        label = row.label
        
        pillow_image = Image.open(self.path+image_id)
        image = np.array(pillow_image)
        
        if self.transform:
            image = self.transform(image=image)['image']

        image = image.astype(np.float32)
        image /= 255
        image = image.transpose(2, 0, 1)
            
        return image, label

    def get_image_to_show(self, image):
        return image.transpose(1, 2, 0)