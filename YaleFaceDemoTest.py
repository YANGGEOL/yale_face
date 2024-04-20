import argparse

import torch
from torchvision import transforms

import os
import cv2

from YaleFaceDataSet import YaleFaceDataset, LoadYaleFaceData, getYaleFaceDataLoader


def LoadTestImageData(img_path, transform):
    if os.path.isdir(img_path):
        images = []
        img_list = os.listdir(img_path)
        for img in img_list:
            image = cv2.imread(f"./{img_path}/{img}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(transform(image))
        return images
    else:
        image = cv2.imread(f"./{img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return transform(image)


def FaceTest():
    # 导入参数
    parser = argparse.ArgumentParser(description="Recognize using CNN")
    parser.add_argument("--imgPath", type=str, default=None, help="TestFace img path")
    args = vars(parser.parse_args())

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150, 150)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载数据
    # 数据集数据
    _, test_loader = getYaleFaceDataLoader()

    test_img = LoadTestImageData(args["imgPath"], transform)

    # 加载模型
    model = torch.load("./model/model.pt")

    # 测试
    model.eval()

    # 判断文件夹
    if isinstance(test_img, list):
        for item in test_img:
            output = model(torch.unsqueeze(item, dim=1))
            probabilities = torch.softmax(output, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_label].item()
            if confidence > 0.90:
                print(f"是第{predicted_label + 1}号人")
            else:
                print("不属于该数据库")
    else:
        # 判断单个文件
        output = model(torch.unsqueeze(test_img, dim=1))
        probabilities = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_label].item()

        if confidence > 0.90:
            print(f"是第{predicted_label + 1}号人")
        else:
            print("不属于该数据库")


if __name__ == '__main__':
    FaceTest()



