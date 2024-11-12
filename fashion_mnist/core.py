import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.onnx import dynamo_export
from utils.nn_utils import get_conv_output_size, get_max_pool_output_size


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 1
        self.conv1_out = 4
        self.conv2_out = 4
        self.kernel_size = 5
        self.pool_kernel_size = 2
        self.fc1_out = 64
        self.fc2_out = 10

        # expected height & width
        self.input_height = 28
        self.input_width = 28

        # calculating input of first fc layer
        self.flatten_conv1_out, self.conv1_output = get_conv_output_size(self.conv1_out, (self.input_height, self.input_width), self.kernel_size)
        self.flatten_pool1_out, self.pool1_output = get_max_pool_output_size(self.conv1_out, self.conv1_output, self.pool_kernel_size)
        self.flatten_conv2_out, self.conv2_output = get_conv_output_size(self.conv2_out,  self.pool1_output, self.kernel_size)
        self.flatten_pool2_out, self.pool2_output = get_max_pool_output_size(self.conv2_out,  self.conv2_output, self.pool_kernel_size)

        self.conv1 = nn.Conv2d(self.in_channels, self.conv1_out, self.kernel_size)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(self.pool_kernel_size)
        self.conv2 = nn.Conv2d(self.conv1_out, self.conv2_out, self.kernel_size)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(self.pool_kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_pool2_out, self.fc1_out)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)

    @property
    def name(self):
        return f'conv1:{self.conv1_out};conv2:{self.conv2_out};fc:{self.fc1_out};kernel:{self.kernel_size}'

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        logits = self.fc2(x)
        return logits


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
    model = NeuralNetwork()
    learning_rate = 0.015421670225292113
    epochs = 5
    batch_size = 8

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

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    print(model)

    train = False
    save_onnx = False
    parse_to_c = True

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

    if parse_to_c:
        from utils.c_parser import CParser
        c_parser = CParser()
        model_c_str = c_parser.model(model, input_size=(28, 28))

        torch_input = torch.randn(1, 1, 28, 28)
        model_c_str = c_parser.tensor_to_const_array('input', torch_input) + model_c_str

        output = model(torch_input)

        model_c_str += '\n' + c_parser.output_testing(c_parser.flatten(output).size(dim=0), output)
        print(model_c_str)

    if save_onnx:
        torch_input = torch.randn(1, 1, 28, 28)
        torch.onnx.export(model, torch_input, "fashion_mnist.onnx")
        # onnx_program = torch.onnx.dynamo_export(model, torch_input)
        # onnx_program.save("fashion_mnist.onnx")
