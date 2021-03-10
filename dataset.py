import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms as T

from config import GlobalConfig

# Use mean and std for pretrained models
# https://pytorch.org/docs/stable/torchvision/models.html

def get_train_transform(img_size):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_valid_transform(img_size):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
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

        self.image_ids = df.StudyInstanceUID
        self.labels = df[GlobalConfig.target_columns]
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        image_id = self.image_ids.iloc[index]
        label = self.labels.iloc[index].values.astype(float)
        
        image = Image.open(self.path+image_id+'.jpg')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_image_to_show(self, image):
        return np.array(T.ToPILImage()(image))
