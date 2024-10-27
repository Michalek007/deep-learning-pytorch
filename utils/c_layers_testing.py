import torch
from torch import nn
import torch.nn.functional as F

from c_parser import CParser
from nn_utils import get_conv_output_size, get_max_pool_output_size


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


def conv(in_tensor: torch.Tensor, in_channels: int, out_channels: int, input_width: int, input_height: int, kernel_width: int, kernel_height: int, stride: int, padding: int):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding)
    params = conv.state_dict()
    out_tensor = conv(in_tensor)

    input_params_dict = {
        'inputChannels': ('size_t', in_channels),
        'inputWidth': ('size_t', input_width),
        'inputHeight': ('size_t', input_height),
        'kernelWidth': ('size_t', kernel_width),
        'kernelHeight': ('size_t', kernel_height),
        'outputChannels': ('size_t', out_channels),
        'stride': ('int', stride),
        'padding': ('int', padding),
    }

    variables = c_parser.dict_to_variables(input_params_dict)
    in_array = c_parser.tensor_to_array('input', in_tensor)
    weights_array = c_parser.tensor_to_array('weights', params['weight'])
    biases_array = c_parser.tensor_to_array('biases', params['bias'])
    # output_size = ((input_width + 2 * padding - kernel_width)/stride + 1) * ((input_height + 2 * padding - kernel_height)/stride + 1) * out_channels
    # output_size = int(output_size)
    output_size, _, _ = get_conv_output_size(out_channels, input_width, input_height, kernel_height, stride, padding)

    print(variables)
    print(in_array)
    print(weights_array)
    print(biases_array)
    print(c_parser.conv(output_size))
    print(c_parser.output_testing(output_size, out_tensor))
    print('\n\n')


def max_pool_default(in_tensor: torch.Tensor, in_channels: int, input_width: int, input_height: int, kernel: int):
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
    # output_size = ((input_width + - kernel)/kernel + 1) * ((input_height - kernel)/kernel + 1) * in_channels
    # output_size = int(output_size)
    output_size, _, _ = get_max_pool_output_size(in_channels, input_width, input_height, kernel)

    print(variables)
    print(in_array)
    print(c_parser.max_pool_default(output_size))
    print(c_parser.output_testing(output_size, out_tensor))
    print('\n\n')


if __name__ == '__main__':
    # in_tensor = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=torch.float)
    # fc(in_tensor, 12, 8)

    # in_tensor = torch.tensor([[[1, 2], [1, 2]]], dtype=torch.float)
    # conv(in_tensor, 1, 2, 2, 2, 2, 2, 1, 0)
    # in_tensor = torch.tensor([[[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]]], dtype=torch.float)
    # conv(in_tensor, 3, 2, 2, 2, 2, 2, 1, 0)
    # conv(in_tensor, 3, 2, 2, 2, 2, 1, 1, 0)

    # in_tensor = torch.tensor([[[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]]], dtype=torch.float)
    # conv(in_tensor, 3, 2, 2, 2, 2, 1, 1, 0)
    # max_pool_default(in_tensor, 3, 2, 2, 2)

    # in_tensor = torch.tensor([[
    #     [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
    #     [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
    #     [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    # ]], dtype=torch.float)
    # max_pool_default(in_tensor, 3, 8, 8, 2)

    # in_tensor = torch.tensor([[
    #     [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    #     [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    #     [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    # ]], dtype=torch.float)
    # max_pool_default(in_tensor, 3, 4, 4, 2)
    pass
