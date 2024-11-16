import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from lwf_dataset import LWFDataset
from model import NeuralNetwork
import numpy as np


training_data = LWFDataset(
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


model = NeuralNetwork()
learning_rate = 0.015421670225292113
epochs = 5
batch_size = 8

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    for batch, (anchor, positive, negative) in enumerate(dataloader):
        # print(batch)
        # print(anchor.size())
        # print(positive.size())
        # print(negative.size())
        # Compute prediction and loss
        anchor_pred = model(anchor)
        positive_pred = model(positive)
        negative_pred = model(negative)
        loss = loss_fn(anchor_pred, positive_pred, negative_pred)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(anchor)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    print(num_batches)
    print(size)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        # for anchor, positive, negative in dataloader:
        #     anchor_pred = model(anchor)
        #     positive_pred = model(positive)
        #     negative_pred = model(negative)
        #     test_loss = loss_fn(anchor_pred, positive_pred, negative_pred)
        #     distance_pos = (anchor_pred - positive_pred).norm().item()
        #     distance_neg = (anchor_pred - negative_pred).norm().item()
        #     print(distance_pos)
        #     print(distance_neg)
        #     if distance_neg > distance_pos:
        #         print('git')
        #         correct += 1
        j = 0
        lim = 500
        for img1, img2, target in dataloader:
            print(img1.size())
            img1, img2 = model(img1), model(img2)
            distance = (img2 - img1).norm(dim=1)
            correct += sum(map(lambda value: 1 if value.item() < 1 else 0, distance))
            # tensor with 1 (value <1) or 0 (value >= 1)
            # dis_tensor - target
            for i in range(len(distance)):
                # distance = diff[i].norm().item()
                distance_item = distance[i].item()
                target_item = target[i].item()
                if target_item == 1 and distance_item < 1:
                    correct += 1
                elif target_item == 0 and distance_item >= 1:
                    correct += 1
                # print(distance_item)
                # print(target_item)
                # print()
                j += 1
                print(j)
            # if j == lim:
            #     break

    print(j)
    # test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


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


if __name__ == "__main__":
    train = False
    save_onnx = False
    parse_to_c = False
    if train:
        loss_fn = nn.TripletMarginLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        correct = 0
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            correct = test_loop(test_dataloader, model, loss_fn)
        print("Done!")

        with open('results.txt', 'a') as f:
            f.write(f'{model.name} lr:{learning_rate};batch_size:{batch_size};epochs:{epochs}\n')
            f.write(str(model))
            f.write(f'Accuracy: {(100*correct):>0.1f}%')
            f.write('\n\n')

        torch.save(model.state_dict(), "lwf.pth")

    if not train:
        model.load_state_dict(torch.load('lwf.pth', weights_only=True))
        model.eval()

    if parse_to_c:
        from utils.c_parser import CParser
        c_parser = CParser()
        model_c_str = c_parser.from_model(model)

        torch_input = torch.randn(1, 1, 28, 28)
        model_c_str = c_parser.to_array('input', torch_input) + model_c_str

        output = model(torch_input)

        print(model_c_str)
        print(c_parser.to_array('expectedOutput', output))

    if save_onnx:
        torch_input = torch.randn(1, 1, 28, 28)
        onnx_program = torch.onnx.dynamo_export(model, torch_input)
        onnx_program.save("lwf.onnx")

    # display_test_results()
    loss_fn = nn.TripletMarginLoss()
    test_loop(test_dataloader, model, loss_fn)
