import torch
from torch.nn.functional import nll_loss

import numpy as np


def FaceTrain(model, train_loader, optimizer, num_epochs, args, device):
    model.train()
    for i in range(num_epochs):
        loss_data = []
        for images, labels in train_loader:
            # images = images.to(device)
            # labels = labels.to(device)

            optimizer.zero_grad()

            output = model(images)

            loss = nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            loss_data.append(loss.data)

        print('Train Epoch: {} \tLoss: {:.6f}'.format(i, np.mean(loss_data)))

    if args["saved"]:
        torch.save(model, './model/model.pt')


def FaceTest(model, test_loader, device):
    model.eval()
    validation_loss = 0
    correct = 0

    for image, label in test_loader:
        output = model(image)
        validation_loss += torch.nn.functional.nll_loss(output, label, size_average=False)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    validation_loss /= len(test_loader.dataset)
    print('\n' + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(validation_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))



