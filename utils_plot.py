import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set()

def show_train_info(train_info):
    _, axes = plt.subplots(2, 2, figsize=(15,10))

    axes[0,0].set_title("Train loss")
    axes[0,0].plot(train_info['train_loss_history'])

    axes[0,1].set_title("Valid loss")
    axes[0,1].plot(train_info['valid_loss_history'])

    axes[1,0].set_title("Train acc")
    axes[1,0].plot(train_info['train_acc_history'])

    axes[1,1].set_title("Valid acc")
    axes[1,1].plot(train_info['valid_acc_history'])

def show_train_info_epoch(train_info, params):
    _, axes = plt.subplots(1, 2, figsize=(17,6))

    x = np.arange(params['num_epoch'])

    axes[0].set_title("Loss")
    axes[0].plot(train_info['train_loss_epochs'], '-o')
    axes[0].plot(train_info['valid_loss_epochs'], '-o')
    axes[0].set_xticks(x)
    axes[0].legend(['train', 'val'], loc='upper right')

    axes[1].set_title("Acc")
    axes[1].plot(train_info['train_acc_epochs'], '-o')
    axes[1].plot(train_info['valid_acc_epochs'], '-o')
    axes[1].set_xticks(x)
    axes[1].legend(['train', 'val'], loc='lower right')


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