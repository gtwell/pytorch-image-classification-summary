import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import argparse


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyImageFolder(Dataset):
    def __init__(self, csv_file, root, transform=None, loader=default_loader):
        self.files = pd.read_csv(csv_file, names=['images', 'labels'])
        # file = file.sample(frac=1).reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.files.iloc[index,0]
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            img = self.transform(img)
        labels = self.files.iloc[index, 1]
        return img, labels

    def __len__(self):
        return len(self.files)


def create_dataset(csv_file = '{}.csv',
                   root_dir = './',
                   phase = ['train', 'val'],
                   shuffle=True,
                   img_size=224,
                   batch_size=32):
    """Create dataset, dataloader for train and test
    Args: label_type (str): Type of label
        csv_file (str): CSV file pattern for file indices.
        root_dir (str): Root dir based on paths in csv file.
        phase: list of str 'train' or 'test'.
    Returns:
        out (dict): A dict contains image_datasets, dataloaders,
            dataset_sizes
    """

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    for x in phase:    # ['train', 'val']
        image_datasets[x] = MyImageFolder(csv_file.format(x),
                                          root_dir,
                                          data_transforms[x])
        if x == 'train':
            dataloaders[x] = DataLoader(image_datasets[x],
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=3)
        else:
            dataloaders[x] = DataLoader(image_datasets[x],
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=3)
        dataset_sizes[x] = len(image_datasets[x])

    out = {'image_datasets': image_datasets,
           'dataloaders': dataloaders,
           'dataset_sizes': dataset_sizes}
    return out


if __name__ == '__main__':
    csv_file = '{}.csv'
    root_dir = './'
    parser = argparse.ArgumentParser("create dataloaders")
    parser.add_argument("--csv_file", type=str, default=csv_file, help="input csv file")
    parser.add_argument("--root_dir", type=str, default=root_dir, help="the files we stored the Images")
    args = parser.parse_args()
    out = create_dataset(csv_file=args.csv_file, root_dir=args.root_dir,)
    image_datasets = out['image_datasets']['train'][0]
    dataloaders = out['dataloaders']
    dataset_sizes = out['dataset_sizes']
    print(image_datasets)
    print(dataloaders)
    print(dataset_sizes)
