import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms as T

# Use mean and std for pretrained models
# https://pytorch.org/docs/stable/torchvision/models.html

def get_train_transform():
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_valid_transform():
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
    ])
    return transform

class ImageDataset(Dataset):
    def __init__(self, df, path, transform=None):
        print('Init dataset T')
        self.df = df
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row.image_id
        label = row.label
        
        image = Image.open(self.path+image_id)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_image_to_show(self, image):
        return np.array(T.ToPILImage()(image))
