import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import argparse

from .train import *
from .dataset import *
from .utils import *
from .model import *

# from train import *
# from dataset import *
# from utils import *
# from model import *

def run(
    path_to_data='/',
    batch_size=32,
    reduce_train=False,
    train_number=0,
    valid_number=0,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10
    ):
    print_version()

    train_df_all = pd.read_csv(path_to_data + 'train.csv')
    print(f'Dataset size: {train_df_all.shape}')

    # train_df = train_df_all
    train_df = train_df_all.sample(frac=1).reset_index(drop=True)

    train_df, valid_df = train_test_split(train_df, test_size=0.2)
    print(f'Dataset size, train: {len(train_df)}, valid: {len(valid_df)}')

    if reduce_train:
        train_df = train_df.sample(frac=1).reset_index(drop=True).head(train_number)
        valid_df = valid_df.sample(frac=1).reset_index(drop=True).head(valid_number)
        print(f'Reduce dataset size, train: {len(train_df)}, valid: {len(valid_df)}')

    img_path = path_to_data + '/train_images/'
    train_dataset, valid_dataset = create_datasets(train_df, valid_df, img_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print(f'DataLoader size, train: {len(train_loader)}, valid: {len(valid_loader)}')

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    model = SimpleModel()
    model.to(device)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print('')
    print('Start training...')
    train_info = train_model(model, device, train_loader, valid_loader, loss, optimizer, num_epoch)

    return train_info, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", type=str, default='/cassava/')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    # seed_everything(args.seed)

    train_info = run(args.path_to_data, args.batch_size, args.num_epoch)

    
