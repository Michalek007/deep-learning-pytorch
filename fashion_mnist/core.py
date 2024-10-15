import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.onnx import dynamo_export
# import torch.onnx.dynamo_export


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        conv1_out = 4
        conv2_out = 4
        fc1_out = 64
        kernel_size = 5
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4*4*conv2_out, fc1_out),
            nn.ReLU(),
            nn.Linear(fc1_out, 10),
        )
        self.name = f'conv1:{conv1_out};conv2:{conv2_out};fc:{fc1_out};kernel:{kernel_size}'

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


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
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


if __name__ == "__main__":
    train = False
    if train:
        loss_fn = nn.CrossEntropyLoss()
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

        torch.save(model.state_dict(), "fashion_mnist.pth")

    if not train:
        model.load_state_dict(torch.load('fashion_mnist.pth', weights_only=True))
        model.eval()

    torch_input = torch.randn(1, 1, 28, 28)
    onnx_program = torch.onnx.dynamo_export(model, torch_input)
    onnx_program.save("fashion_mnist.onnx")
