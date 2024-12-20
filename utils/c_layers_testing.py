import torch
from torch import nn
import torch.nn.functional as F
from typing import Union
from torchvision.ops.boxes import nms
from torchvision.ops.boxes import box_iou

from c_parser import CParser
from nn_utils import get_conv_output_size, get_max_pool_output_size, params_to_tuple


c_parser = CParser()


def fc(in_tensor: torch.Tensor, input_len: int, output_len: int):
    input_params_dict = {
        'inputLen': ('size_t', input_len),
        'outputLen': ('size_t', output_len),
    }
    fc = nn.Linear(input_len, output_len)
    params = fc.state_dict()
    out_tensor = fc(in_tensor)

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_array('input', in_tensor)
    weights_array = c_parser.tensor_to_array('weights', params['weight'])
    biases_array = c_parser.tensor_to_array('biases', params['bias'])

    print(variables)
    print(in_array)
    print(weights_array)
    print(biases_array)
    print(c_parser.fc())
    print(c_parser.output_testing(output_len, out_tensor))
    print('\n\n')


def conv_(in_tensor: torch.Tensor, in_channels: int, out_channels: int, input_height: int, input_width: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0):
    kernel_size, stride, padding = params_to_tuple(kernel_size, stride, padding)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    params = conv.state_dict()
    out_tensor = conv(in_tensor)

    input_params_dict = {
        'inputChannels': ('size_t', in_channels),
        'inputHeight': ('size_t', input_height),
        'inputWidth': ('size_t', input_width),
        'kernelHeight': ('size_t', kernel_size[0]),
        'kernelWidth': ('size_t', kernel_size[1]),
        'outputChannels': ('size_t', out_channels),
        'strideH': ('int', stride[0]),
        'strideW': ('int', stride[1]),
        'paddingH': ('int', padding[0]),
        'paddingW': ('int', padding[1])
    }

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_array('input', in_tensor)
    weights_array = c_parser.tensor_to_array('weights', params['weight'])
    biases_array = c_parser.tensor_to_array('biases', params['bias'])
    output_size, _ = get_conv_output_size(out_channels, (input_height, input_width), kernel_size, stride, padding)

    print(variables)
    print(in_array)
    print(weights_array)
    print(biases_array)
    print(c_parser.conv_(output_size))
    print(c_parser.output_testing(output_size, out_tensor))
    print('\n\n')


def conv(in_tensor: torch.Tensor, in_channels: int, out_channels: int, input_height: int, input_width: int, kernel_height: int, kernel_width: int, stride: int = 1, padding: int = 0):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding)
    params = conv.state_dict()
    out_tensor = conv(in_tensor)

    input_params_dict = {
        'inputChannels': ('size_t', in_channels),
        'inputHeight': ('size_t', input_height),
        'inputWidth': ('size_t', input_width),
        'kernelHeight': ('size_t', kernel_height),
        'kernelWidth': ('size_t', kernel_width),
        'outputChannels': ('size_t', out_channels),
        'stride': ('int', stride),
        'padding': ('int', padding),
    }

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_array('input', in_tensor)
    weights_array = c_parser.tensor_to_array('weights', params['weight'])
    biases_array = c_parser.tensor_to_array('biases', params['bias'])
    output_size, _ = get_conv_output_size(out_channels, (input_height, input_width), (kernel_height, kernel_width), stride, padding)

    print(variables)
    print(in_array)
    print(weights_array)
    print(biases_array)
    print(c_parser.conv(output_size))
    print(c_parser.output_testing(output_size, out_tensor))
    print('\n\n')


def max_pool_default(in_tensor: torch.Tensor, in_channels: int, input_height: int, input_width: int, kernel: int):
    pool = nn.MaxPool2d(kernel, kernel)
    out_tensor = pool(in_tensor)

    input_params_dict = {
        'inputChannels': ('size_t', in_channels),
        'inputWidth': ('size_t', input_width),
        'inputHeight': ('size_t', input_height),
        'kernel': ('size_t', kernel),
    }

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_array('input', in_tensor)
    output_size, _ = get_max_pool_output_size(in_channels, (input_height, input_width), kernel)

    print(variables)
    print(in_array)
    print(c_parser.max_pool_default(output_size))
    print(c_parser.output_testing(output_size, out_tensor))
    print('\n\n')


def max_pool(in_tensor: torch.Tensor, in_channels: int, input_height: int, input_width: int, kernel_height: int, kernel_width: int, stride: int, padding: int = 0):
    pool = nn.MaxPool2d(kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding)
    out_tensor = pool(in_tensor)

    input_params_dict = {
        'inputChannels': ('size_t', in_channels),
        'inputHeight': ('size_t', input_height),
        'inputWidth': ('size_t', input_width),
        'kernelHeight': ('size_t', kernel_height),
        'kernelWidth': ('size_t', kernel_width),
        'stride': ('int', stride),
        'padding': ('int', padding),
    }

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_array('input', in_tensor)
    output_size, _ = get_max_pool_output_size(in_channels, (input_height, input_width), (kernel_height, kernel_width), stride, padding)

    print(variables)
    print(in_array)
    print(c_parser.max_pool(output_size))
    print(c_parser.output_testing(output_size, out_tensor))
    print('\n\n')


def max_pool_(in_tensor: torch.Tensor, in_channels: int, input_height: int, input_width: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, ceil_mode: bool = False):
    kernel_size, stride, padding = params_to_tuple(kernel_size, stride, padding)
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
    out_tensor = pool(in_tensor)

    input_params_dict = {
        'inputChannels': ('size_t', in_channels),
        'inputHeight': ('size_t', input_height),
        'inputWidth': ('size_t', input_width),
        'kernelHeight': ('size_t', kernel_size[0]),
        'kernelWidth': ('size_t', kernel_size[1]),
        'strideH': ('int', stride[0]),
        'strideW': ('int', stride[1]),
        'paddingH': ('int', padding[0]),
        'paddingW': ('int', padding[1]),
        'ceilMode': ('int', int(ceil_mode))
    }

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_array('input', in_tensor)
    output_size, _ = get_max_pool_output_size(in_channels, (input_height, input_width), kernel_size, stride, padding, ceil_mode)

    print(variables)
    print(in_array)
    print(c_parser.max_pool_(output_size))
    print(c_parser.output_testing(output_size, out_tensor))
    print('\n\n')


def prelu(in_tensor: torch.Tensor, in_channels: int, input_height: int, input_width: int):
    input_params_dict = {
        'inputChannels': ('size_t', in_channels),
        'inputHeight': ('size_t', input_height),
        'inputWidth': ('size_t', input_width),
    }
    output_len = in_channels * input_height * input_width
    prelu = nn.PReLU(in_channels)
    params = prelu.state_dict()
    out_tensor = prelu(in_tensor)
    print(out_tensor)

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_const_array('input', in_tensor)
    weights_array = c_parser.tensor_to_const_array('weights', params['weight'])

    print(variables)
    print(in_array)
    print(weights_array)
    print(c_parser.prelu(output_len))
    print(c_parser.output_testing(output_len, out_tensor))
    print('\n\n')


def softmax(in_tensor: torch.Tensor, input_len: int):
    input_params_dict = {
        'inputLen': ('size_t', input_len)
    }
    softmax = nn.Softmax(dim=1)
    out_tensor = softmax(in_tensor)

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_const_array('input', in_tensor)

    print(variables)
    print(in_array)
    print(c_parser.softmax(input_len))
    print(c_parser.output_testing(input_len, out_tensor))
    print('\n\n')


def softmax2d(in_tensor: torch.Tensor, in_channels: int, input_height: int, input_width: int, dim: int):
    input_params_dict = {
        'inputChannels': ('size_t', in_channels),
        'inputHeight': ('size_t', input_height),
        'inputWidth': ('size_t', input_width),
        'dim': ('size_t', dim-1),
    }
    output_len = in_channels * input_height * input_width
    softmax = nn.Softmax(dim=dim)
    out_tensor = softmax(in_tensor)
    print(out_tensor)

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_const_array('input', in_tensor)

    print(variables)
    print(in_array)
    print(c_parser.softmax2d(output_len))
    print(c_parser.output_testing(output_len, out_tensor))
    print('\n\n')


if __name__ == '__main__':
    # in_tensor = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=torch.float)
    # fc(in_tensor, 12, 8)

    # in_tensor = torch.tensor([[[1, 2], [1, 2]]], dtype=torch.float)
    # conv(in_tensor, 1, 2, 2, 2, 2, 2)
    # in_tensor = torch.tensor([[[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]]], dtype=torch.float)
    # conv(in_tensor, 3, 2, 2, 2, 2, 2)

    # in_tensor = torch.tensor([[[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]]], dtype=torch.float)
    # conv(in_tensor, in_channels=3, out_channels=2, input_height=2, input_width=2, kernel_height=2, kernel_width=1)
    # max_pool_default(in_tensor, in_channels=3, input_height=2, input_width=2, kernel=2)

    # in_tensor = torch.tensor([[
    #     [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
    #     [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
    #     [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    # ]], dtype=torch.float)
    # max_pool_default(in_tensor, in_channels=3, input_height=8, input_width=8, kernel=2)

    # in_tensor = torch.tensor([[
    #     [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    #     [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    #     [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    # ]], dtype=torch.float)
    # max_pool_default(in_tensor, in_channels=3, input_height=4, input_width=4, kernel=2)

    # in_tensor = torch.tensor([[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]], dtype=torch.float)
    # conv(in_tensor, in_channels=1, out_channels=2, input_height=2,input_width= 6, kernel_height=2, kernel_width=2, stride=2)
    # max_pool(in_tensor, in_channels=1, input_height=2, input_width= 6, kernel_height=2, kernel_width=4, stride=2)

    # in_tensor = torch.tensor([[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]], dtype=torch.float)
    # conv(in_tensor, in_channels=1, out_channels=2, input_height=2,input_width= 6, kernel_height=2, kernel_width=2, stride=2, padding=1)
    # max_pool(in_tensor, in_channels=1, input_height=2, input_width=6, kernel_height=2, kernel_width=4, stride=2, padding=1)

    # in_tensor = torch.tensor([[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]], dtype=torch.float)
    # conv_(in_tensor, in_channels=1, out_channels=2, input_height=4, input_width=6, kernel_size=2, stride=(1, 2), padding=0)
    # max_pool_(in_tensor, in_channels=1, input_height=4, input_width=6, kernel_size=2, stride=(1, 2), padding=0)

    # in_tensor = torch.tensor([[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]], dtype=torch.float)
    # conv_(in_tensor, in_channels=1, out_channels=2, input_height=2, input_width=6, kernel_size=2, stride=2, padding=(0, 1))
    # max_pool_(in_tensor, in_channels=1, input_height=2, input_width=6, kernel_size=2, stride=2, padding=(0, 1))

    # in_tensor = torch.tensor([
    #     [[1, 2, 3, -4, -5, -6], [-1, -2, -3, 4, 5, 6]],
    #     [[0, 0, 0, 4, 5, 6], [1, 2, 3, -4, -5, -6]]], dtype=torch.float)
    # prelu(in_tensor, 2, 2, 6)
    # softmax2d(in_tensor, 2, 2, 6, dim=1)
    # softmax2d(in_tensor, 2, 2, 6, dim=2)

    # in_tensor = torch.tensor([[1, 2, 3, -4, -5, -6, -1, -2, -3, 4, 5, 6]], dtype=torch.float)
    # softmax(in_tensor, in_tensor.size(1))

    # in_tensor = torch.tensor([[
    #     [[2, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    #     [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    #     [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    # ]], dtype=torch.float)
    # softmax2d(in_tensor, 3, 4, 4, dim=1)

    # in_tensor = torch.tensor([[
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3]
    # ]], dtype=torch.float)
    # max_pool_(in_tensor, in_channels=1, input_height=3, input_width=3, kernel_size=2, stride=2, padding=(0, 0), ceil_mode=True)
    # max_pool_(in_tensor, in_channels=1, input_height=3, input_width=3, kernel_size=3, stride=3, padding=(1, 1), ceil_mode=True)

    # in_tensor = torch.tensor([
    #     [[1, 2, 3], [4, 5, 6]],
    #     [[7, 8, 9], [10, 11, 12]],
    # ], dtype=torch.float)
    # print(c_parser.tensor_to_const_array('input', in_tensor))
    # in_tensor = in_tensor.permute(2, 1, 0)
    # print(c_parser.output_testing(2*2*3, in_tensor))
    # in_tensor = in_tensor.permute(2, 0, 1)
    # print(c_parser.output_testing(2*2*3, in_tensor))

    # in_tensor = torch.torch.randn(3, 12, 12)
    # print(c_parser.tensor_to_const_array('input', in_tensor))
    # in_tensor = in_tensor.permute(2, 1, 0)
    # print(c_parser.output_testing(3*12*12, in_tensor))

    # in_tensor = torch.rand(1, 3, 4, 4)
    # max_pool_(in_tensor, in_channels=3, input_height=4, input_width=4, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    # in_tensor = torch.rand(1, 3, 4, 4) * -1
    # max_pool_(in_tensor, in_channels=3, input_height=4, input_width=4, kernel_size=3, stride=2, padding=0, ceil_mode=True)

    # conv_(in_tensor, in_channels=3, out_channels=2, input_height=4, input_width=4, kernel_size=2, stride=2, padding=(2, 2))

    # x = (
    #     (50, 50, 70, 70),
    #     (10, 10, 20, 20),
    # )
    # y = (
    #     (40, 40, 70, 70),
    #     (60, 60, 80, 80),
    # )
    # x = torch.tensor(x, dtype=torch.float)
    # y = torch.tensor(y, dtype=torch.float)
    # print(c_parser.tensor_to_const_array('boxes', x))
    # print(c_parser.tensor_to_const_array('boxes2', y))
    # print(c_parser.output_testing(len(x)*len(y), box_iou(x, y)))

    # x = (
    #     (50, 50, 70, 70),
    #     (10, 10, 20, 20),
    #     (40, 40, 70, 70),
    #     (60, 60, 80, 80),
    # )
    # scores = (0.8, 0.1, 0.1, 0.1)
    # x = torch.tensor(x, dtype=torch.float)
    # scores = torch.tensor(scores, dtype=torch.float)
    # indexes = nms(x, scores, 0.3)
    # x_after_nms = x[indexes]
    # print(c_parser.tensor_to_const_array('boxes', x))
    # print(c_parser.tensor_to_const_array('scores', scores))
    # print(c_parser.output_testing(x_after_nms.flatten().size(0), x_after_nms))


    # print(c_parser.tensor_to_const_array('boxes', x))
    # print(c_parser.tensor_to_const_array('boxes2', y))
    # print(c_parser.output_testing(len(x)*len(y), box_iou(x, y)))

    # def torch_pool(inputs, target_size):
    #     start_points = (torch.arange(target_size, dtype=torch.float32) * (inputs.size(-1) / target_size)).long()
    #     end_points = ((torch.arange(target_size, dtype=torch.float32) + 1) * (
    #                 inputs.size(-1) / target_size)).ceil().long()
    #     pooled = []
    #     for idx in range(target_size):
    #         pooled.append(torch.mean(inputs[:, :, start_points[idx]:end_points[idx]], dim=-1, keepdim=False))
    #     pooled = torch.cat(pooled, -1)
    #     return pooled
    #
    # x = torch_pool(torch.tensor(inps_torch), 4)
    # print(x)

    x = torch.rand((3, 8, 10))
    y = nn.AdaptiveAvgPool2d((4, 4))(x)
    print(c_parser.tensor_to_const_array('input', x))
    print(c_parser.output_testing(3*4*4, y))

    input_size = torch.tensor((8, 10))
    output_size = torch.tensor((4, 4))
    kernel_size = (input_size + output_size - 1) // output_size
    print(kernel_size)

    print((8+4-1) // 4)
    pass
