import shutil
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
import torch
import torch.utils.data
from torchvision import datasets, transforms


def load_data(chat_file):
    """
    Loads the data from the file system
    :param chat_file: path to WhatsApp _chat.txt file
    :return: data frame of parsed chats
    """
    with open(chat_file) as f:
        content = f.readlines()

    str = "\n".join(content)
    m = re.compile("\[([^\]]*)\] ([^:]*):.*< pièce jointe : ([^ ]*) >").findall(
        str)

    df = pd.DataFrame(m, columns=["date", "from", "photo"])

    df["date"] = pd.to_datetime(df["date"])

    # some data validation: filter out some unneeded entries

    # - first entry not needed, just some icon change
    df = df.drop(0)

    # - some entries are videos, we only look at the JPGs
    df = df[df["photo"].apply(lambda x: x.endswith(".jpg"))]

    return df


# %%

def show_plots(df):
    """
    Show some plots of chat data
    :param df: data frame of WhatsApp chat data
    :return:
    """
    df.groupby(['from']).count()["photo"].plot(kind='bar')
    plt.title("Top reporters")
    plt.tight_layout()
    plt.show()

    df.groupby(df['date'].dt.month).count()["photo"].plot(kind='bar')
    plt.title("Photos per month")
    plt.xlabel("month")
    plt.show()

    df.groupby(df['date'].dt.hour).count()["photo"].plot(kind='bar')
    plt.title("Photos per hour")
    plt.xlabel("hour of the day")
    plt.show()

    hours = df.groupby(['from', df['date'].dt.hour]).count()["photo"]
    for k in df['from'].unique():
        plt.plot(hours[k].index, hours[k].values, label=k)
    plt.legend()
    plt.xlabel("hour of the day")
    plt.show()


def prepare_loaders(data_dir, validation_split=0.2, random_seed=132,
                    batch_size=4):
    """
    Prepares the PyTorch loaders
    :param data_dir: where to load photos from
    :param validation_split: percentage used for validation
    :param random_seed: seed to use for data split
    :param batch_size: training and validation batch size
    :return: dict of data loaders, sizes and class names
    """
    # TODO apply separate transforms for validation split
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    indices = list(range(len(dataset)))
    split = int(np.floor(validation_split * len(dataset)))

    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    dataloaders = {
        "train": torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=train_sampler),
        "val": torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=valid_sampler),
        "full_val": torch.utils.data.DataLoader(dataset,
                                                sampler=valid_sampler,
                                                batch_size=1)
    }

    dataset_sizes = {x: batch_size * len(dataloaders[x]) for x in
                     ['train', 'val']}

    class_names = dataset.classes

    return dataloaders, dataset_sizes, class_names


def prepare_fs_structure(df, all_photos_dir, split_photos_dir):
    """
    Moves the photos into a per-class folder structure
    :param df: data frame with chat data
    :param all_photos_dir: path to directory of all photos
    :param split_photos_dir: path to directory where targets should be placed
    :return:
    """
    for reporter, group in df.groupby('from'):
        target_dir = os.path.join(split_photos_dir, reporter)
        Path(target_dir).mkdir(exist_ok=True, parents=True)

        for photo in group["photo"]:
            src = os.path.join(all_photos_dir, photo)
            dst = os.path.join(target_dir, photo)
            shutil.copy(src, dst)
