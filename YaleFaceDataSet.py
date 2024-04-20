import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms


def LoadYaleFaceData():
    data_lst = os.listdir("./yalefaces")

    labels = []
    images = []
    for item in data_lst:
        if not item.endswith("txt"):
            label = item.split(".")[0].split("t")[1]
            image = plt.imread(f"./yalefaces/{item}")
            rgb_pixels = np.stack((image, image, image), axis=2)

            labels.append(int(label) - 1)
            images.append(rgb_pixels)
    return [images, labels]


def getYaleFaceDataLoader():

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150, 150)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = LoadYaleFaceData()
    yale_data = YaleFaceDataset(dataset[0], dataset[1], transform)

    batch_size = 16

    train_dataset, test_dataset = random_split(yale_data, [140, 25])

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset)


class YaleFaceDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)

        return image, label

