import csv
import glob
import os
import random
import time

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RSdataset(Dataset):

    def __init__(self, root: str, resize: int, mode: str):
        super().__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}  # "sq...":0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train':  # 60%
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode == 'val':  # 20% = 60%->80%
            self.images = self.images[int(
                0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(
                0.6*len(self.labels)):int(0.8*len(self.labels))]
        else:  # 20% = 80%->100%
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'RSdataset\\mewtwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # 1167, 'RSdataset\\bulbasaur\\00000000.png'
            #print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'RSdataset\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'RSdataset\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                #print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'RSdataset\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

    def __len__(self):

        return len(self.images)

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'RSdataset\\bulbasaur\\00000000.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize), int(self.resize))),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        #print(img.shape, label)
        return img, label


def main():
    db = RSdataset('WHURS19', 300, 'train')
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)
    for x, y in loader:
        print('sample:', x.shape, y.shape, y)
        time.sleep(10)


if __name__ == '__main__':
    main()
