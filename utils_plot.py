import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set()

def show_dataset_grid(dataset):
    nrow, ncol = 3, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        image, label = dataset[i]
        plt.imshow(dataset.get_image_to_show(image))
        ax.set_title(f'Label: {label}\nShape: {np.array(image).shape}', fontsize=16)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_dataset(dataset, count=5, random=True):    
    size = 5
    plt.figure(figsize=(count*size,size))

    if random:   
        indices = np.random.choice(np.arange(len(dataset)), count, replace=False)
    else:
        indices = np.arange(count)
     
    for i, index in enumerate(indices):    
        image, label = dataset[index]
        plt.subplot(1,count,i+1)
        plt.title(f'Label: {label}\nShape: {np.array(image).shape}', fontsize=16)
        plt.imshow(dataset.get_image_to_show(image))
        plt.grid(False)
        plt.axis('off')

def show_conv_weight(conv):
    weight = conv.weight.data.cpu().numpy()
    weight = weight.transpose(0, 2, 3, 1)
    
    ubound = 255
    nrow, ncol = 8, 8
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        image = weight[i]
        low, high = np.min(image), np.max(image)
        image = ubound * (image - low) / (high - low)
        ax.imshow(image.astype('uint8'))
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()