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
        self.in_channels = 3
        self.conv1_out = 4
        self.conv2_out = 4
        self.kernel_size = 5
        self.kernel_size_2 = 5
        self.max_pool_kernel_size = 2
        self.fc1_out = 64
        self.fc2_out = 64

        input_width = 160
        input_height = 160
        conv1_out, conv1_outputW, conv1_outputH = get_conv_output_size(self.conv1_out, input_width, input_height, self.kernel_size)
        pool1_out, pool1_outputW, pool1_outputH = get_max_pool_output_size(self.conv1_out, conv1_outputW, conv1_outputH, self.max_pool_kernel_size)
        conv2_out, conv2_outputW, conv2_outputH = get_conv_output_size(self.conv2_out,  pool1_outputW, pool1_outputH, self.kernel_size_2)
        pool2_out, pool2_outputW, pool2_outputH = get_max_pool_output_size(self.conv2_out,  conv2_outputW, conv2_outputH, self.max_pool_kernel_size)
        # print(pool2_out)

        self.conv1 = nn.Conv2d(self.in_channels, self.conv1_out, self.kernel_size)
        self.pool = nn.MaxPool2d((self.max_pool_kernel_size, self.max_pool_kernel_size))
        self.conv2 = nn.Conv2d(self.conv1_out, self.conv2_out, self.kernel_size_2)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(pool2_out, self.fc1_out),
            nn.ReLU(),
            nn.Linear(self.fc1_out, self.fc2_out),
        )

    @property
    def name(self):
        return f'conv1:{self.conv1_out};conv2:{self.conv2_out};fc:{self.fc1_out};kernel:{self.kernel_size}'

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        # print(x.size())
        logits = self.linear_relu_stack(x)
        return logits

    def layers(self):
        input_width = 160
        input_height = 160

        conv1_out, conv1_outputW, conv1_outputH = get_conv_output_size(self.conv1_out, input_width, input_height, self.kernel_size)
        pool1_out, pool1_outputW, pool1_outputH = get_max_pool_output_size(self.conv1_out, conv1_outputW, conv1_outputH, self.max_pool_kernel_size)
        conv2_out, conv2_outputW, conv2_outputH = get_conv_output_size(self.conv2_out,  pool1_outputW, pool1_outputH, self.kernel_size_2)
        pool2_out, pool2_outputW, pool2_outputH = get_max_pool_output_size(self.conv2_out,  conv2_outputW, conv2_outputH, self.max_pool_kernel_size)

        # print(conv1_out, conv1_outputW, conv1_outputH)
        # print(pool1_out, pool1_outputW, pool1_outputH)
        # print(conv2_out, conv2_outputW, conv2_outputH)
        # print(pool2_out, pool2_outputW, pool2_outputH)

        return (
            ('conv', {'in_channels': self.in_channels, 'out_channels': self.conv1_out, 'kernel': self.kernel_size, 'input_width': input_width, 'input_height': input_height, 'output_len': conv1_out}),
            ('relu', {}),
            ('max_pool', {'in_channels': self.conv1_out, 'kernel': self.max_pool_kernel_size, 'input_width': conv1_outputW, 'input_height': conv1_outputH, 'output_len': pool1_out}),
            ('conv', {'in_channels': self.conv1_out, 'out_channels': self.conv2_out, 'kernel': self.kernel_size_2, 'input_width': pool1_outputW, 'input_height': pool1_outputH, 'output_len': conv2_out}),
            ('relu', {}),
            ('max_pool', {'in_channels': self.conv2_out, 'kernel': self.max_pool_kernel_size, 'input_width': conv2_outputW, 'input_height': conv2_outputH, 'output_len': pool2_out}),
            ('fc', {'input_len': self.conv2_out*pool2_outputW*pool2_outputH, 'output_len': self.fc1_out}),
            ('relu', {}),
            ('fc', {'input_len': self.fc1_out, 'output_len': self.fc2_out})
        )
