import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math 

# sns.set()

def show_train_info(train_info):
    _, axes = plt.subplots(2, 2, figsize=(15,10))

    axes[0,0].set_title("Train loss")
    axes[0,0].plot(train_info['train_loss_history'])

    axes[0,1].set_title("Valid loss")
    axes[0,1].plot(train_info['valid_loss_history'])

    axes[1,0].set_title("Train score")
    axes[1,0].plot(train_info['train_score_history'])

    axes[1,1].set_title("Valid score")
    axes[1,1].plot(train_info['valid_score_history'])

def show_train_info_epoch(train_info):
    _, axes = plt.subplots(1, 2, figsize=(17,6))

    x = np.arange(len(train_info['train_loss_epochs']))

    axes[0].set_title("Loss")
    axes[0].plot(train_info['train_loss_epochs'], '-o')
    axes[0].plot(train_info['valid_loss_epochs'], '-o')
    axes[0].set_xticks(x)
    axes[0].legend(['train', 'val'], loc='upper right')

    axes[1].set_title("Score")
    axes[1].plot(train_info['train_score_epochs'], '-o')
    axes[1].plot(train_info['valid_score_epochs'], '-o')
    axes[1].set_xticks(x)
    axes[1].legend(['train', 'val'], loc='lower right')

def show_history_epoch(history):
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(history))
    ax.set_xticks(x)
    ax.plot(history, '-o')


def show_dataset_grid(dataset):
    nrow, ncol = 3, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        image, label = dataset[i]
        ax.imshow(dataset.get_image_to_show(image))
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
    num_weight = weight.shape[0]
    
    ubound = 255
    ncol = 8
    nrow = int(num_weight / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        image = weight[i]
        low, high = np.min(image), np.max(image)
        image = ubound * (image - low) / (high - low)
        ax.imshow(image.astype('uint8'))
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# TODO: try
# from sklearn.metrics import plot_confusion_matrix
def show_confusion_matrix(confusion_matrix):
    # plt.figure(figsize=(7,7))
    plt.title("Confusion matrix")

    sns.heatmap(
        confusion_matrix,
        cmap="GnBu",
        fmt="d",
        linewidths=1.5,
        annot=True
    )

    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.show()
    
def show_cms(train_info):
    cm_number = len(train_info['best_cms'])
    ncol = 3
    nrow = math.ceil(cm_number / 3)
    size = 5
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*size, nrow*size))
    fig.suptitle('Confusion matrices')

    axes = axes.flatten()

    for i, cm in enumerate(train_info['best_cms']):
        ax = axes[i]
        score = train_info['best_scores'][i]
        column = train_info['target_columns'][i]
        ax.set_title(f'{column}\nScore: {score:.5f}', fontsize=15)
        sns.heatmap(
            cm,
            cmap="GnBu",
            fmt="d",
            linewidths=1.5,
            annot=True,
            ax=ax
        )   
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Target")

    fig.tight_layout(pad=3.0)