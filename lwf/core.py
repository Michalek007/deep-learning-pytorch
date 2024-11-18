import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from lwf_dataset import LWFDataset
from model import NeuralNetwork
import numpy as np
import logging
import shutil
import os
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename='logs.log', level=logging.INFO)

MARGIN = 0.7
IN_CHANNELS = 3
IMAGE_H = 160
IMAGE_W = 160
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 8
FILENAME = 'lwf.pth'
ONNX_FILENAME = 'lwf.onnx'
CP_DIR_NUMBER = None


def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    dataset_len = len(dataloader.dataset)
    batches_count = len(dataloader)
    running_loss = 0

    for batch, (anchor, positive, negative) in enumerate(dataloader):
        # Compute prediction and loss
        anchor_pred = model(anchor)
        positive_pred = model(positive)
        negative_pred = model(negative)
        loss = loss_fn(anchor_pred, positive_pred, negative_pred)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(anchor)
            print(f"loss: {loss:>8f}  [{current:>5d}/{dataset_len:>5d}]")

    running_loss /= batches_count
    print(f"Train loss: {running_loss:>8f}")
    return running_loss


def test_loop(dataloader, model, loss_fn, margin: float):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    dataset_len = len(dataloader.dataset)
    batches_count = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (img1, img2, target) in enumerate(dataloader):
            img1, img2 = model(img1), model(img2)
            distance = (img2 - img1).norm(dim=1)
            xor = torch.logical_xor(distance >= margin, target)
            correct += xor.sum().item()
            print(f"accuracy: {(100*correct/((batch+1)*BATCH_SIZE)):>0.1f}% [{batch * BATCH_SIZE + len(img1):>5d}/{dataset_len:>5d}]")

    test_loss /= batches_count
    correct /= dataset_len
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def display_test_results():
    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i in range(500):
            img1, img2, img3 = test_data[i]
            img1, img2, img3 = img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0)
            label1, label2, label3 = test_data.triplets_labels[i]

            train_results.append(model(img1).norm().item())
            train_results.append(model(img2).norm().item())
            train_results.append(model(img3).norm().item())
            labels.append(label1)
            labels.append(label2)
            labels.append(label3)

    labels = np.array(labels)
    train_results = np.array(train_results)
    print(labels)
    print(train_results)

    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = train_results[labels == label]
        # print(tmp)
        plt.scatter(tmp, tmp, label=label)

    plt.legend()
    plt.show()


def save_checkpoint(model: nn.Module, optimizer):
    global CP_DIR_NUMBER
    torch.save(model.state_dict(), f'cp_{FILENAME}')
    torch.save(optimizer.state_dict(), f'cp_optimizer_{FILENAME}')
    path = Path(f'models')
    if not path.exists():
        path.mkdir()
    dirs = os.listdir(path)
    if '0' not in dirs:
        path = path.joinpath('0')
        path.mkdir(exist_ok=True)
        CP_DIR_NUMBER = '0'
    else:
        if CP_DIR_NUMBER is None:
            dir_number = sorted(map(int, filter(lambda arg: arg.isnumeric(), dirs)))[-1] + 1
            CP_DIR_NUMBER = str(dir_number)
        path = path.joinpath(CP_DIR_NUMBER)
        path.mkdir(exist_ok=True)
    shutil.copy(f'cp_{FILENAME}', path)
    shutil.copy(f'cp_optimizer_{FILENAME}', path)
    shutil.copy(f'best_model_{FILENAME}', path)


def load_checkpoint(model: nn.Module, optimizer):
    model.load_state_dict(torch.load(f'cp_{FILENAME}', weights_only=True))
    optimizer.load_state_dict(torch.load(f'cp_optimizer_{FILENAME}'))


if __name__ == '__main__':
    train_data = LWFDataset(
        root="data",
        split="train",
        download=True,
        transform=ToTensor()
    )

    test_data = LWFDataset(
        root="data",
        # split="test",
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = NeuralNetwork(IN_CHANNELS, (IMAGE_H, IMAGE_W))
    print(model)

    train = True
    save_onnx = False
    load_from_checkpoint = False
    save_to_checkpoint = True

    if train:
        loss_fn = nn.TripletMarginLoss(margin=MARGIN)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

        if load_from_checkpoint:
            load_checkpoint(model, optimizer)
            optimizer.load_state_dict(torch.load(f'cp_optimizer_{FILENAME}'))

        last_accuracy = 0
        accuracy, average_train_loss, average_test_loss = 0, 0, 0
        logging.info(f"Training of model: f'lr:{LEARNING_RATE};batch_size:{BATCH_SIZE};epochs:{EPOCHS}; {model.name}'")
        for t in range(EPOCHS):
            print(f"Epoch {t + 1}\n-------------------------------")
            average_train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
            accuracy, average_test_loss = test_loop(test_dataloader, model, loss_fn, MARGIN)
            logging.info(f"Accuracy: {(100 * accuracy):>0.1f}%; Average train loss: {average_train_loss:>0.8f}; Average test loss: {average_test_loss:>0.8f} after {t+1} epoch. ")
            if save_to_checkpoint:
                if accuracy > last_accuracy:
                    torch.save(model.state_dict(), f'best_model_{FILENAME}')
                    last_accuracy = accuracy
                save_checkpoint(model, optimizer)
        print("Done!")

        with open('results.txt', 'a') as f:
            f.write(f'lr:{LEARNING_RATE};batch_size:{BATCH_SIZE};epochs:{EPOCHS} {model.name};dir_number:{CP_DIR_NUMBER}\n')
            f.write(str(model))
            f.write(f' Accuracy: {(100 * accuracy):>0.1f}%; Average train loss: {average_train_loss:>0.8f}; Average test loss: {average_test_loss:>0.8f}')
            f.write('\n\n')

        torch.save(model.state_dict(), FILENAME)

    if not train:
        model.load_state_dict(torch.load(FILENAME, weights_only=True))
        model.eval()
        accuracy, average_loss = test_loop(test_dataloader, model, nn.TripletMarginLoss(margin=MARGIN))
        print(f'Accuracy: {(100 * accuracy):>0.1f}%; Average test loss: {average_loss:>0.8f}')

    if save_onnx:
        torch_input = torch.randn(1, IN_CHANNELS, IMAGE_H, IMAGE_W)
        torch.onnx.export(model, torch_input, ONNX_FILENAME)
