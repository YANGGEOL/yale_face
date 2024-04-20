import os
import argparse
from YaleFaceDataSet import getYaleFaceDataLoader
from YaleFaceRecognitionModel import Network
from YaleFaceTrainAndTest import FaceTrain, FaceTest

from torch import optim
import torch


def main():
    # 获取参数 是否训练 是否保存模型
    parser = argparse.ArgumentParser(description="Recognize using CNN")
    parser.add_argument("--train", action='store_true', default=False,  help="Decide whether to train with trainset")
    parser.add_argument("--saved", action='store_true', default=False, help="Save Model Option")
    args = vars(parser.parse_args())

    # 数据集
    train_loader, test_loader = getYaleFaceDataLoader()
    # 训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 看是训练还是测试
    if args['train']:
        model = Network()
        # model.to(device)

        # 超参数
        num_epochs = 50
        lr = 0.01
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练
        FaceTrain(model, train_loader, optimizer, num_epochs, args, device)
    else:
        model = torch.load('./model/model.pt')

        # 测试
        FaceTest(model, test_loader, device)


if __name__ == '__main__':
    main()

