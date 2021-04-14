import argparse

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class SkierDataset(Dataset):
    def __init__(self)
        pass

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass

def get_dataloader(args):
    episodes = list()
    data_dir = Path(args.dataset_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    args = parser.parse_args()
