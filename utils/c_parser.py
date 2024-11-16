import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple

from utils.nn_utils import get_conv_output_size, get_max_pool_output_size


class CParser:
    def __init__(self):
        self.flatten = nn.Flatten(start_dim=0)

    def tensor_to_array(self, name: str, tensor: torch.Tensor):
        tensor = self.flatten(tensor)
        if tensor.dtype == torch.float:
            data_type = 'float'
        elif tensor.dtype == torch.int:
            data_type = 'int'
        else:
            data_type = 'undefined'
        c_array = f'{data_type} {name}[] = {{'

        c_array += ', '.join(map(lambda t: str(round(t.item(), 5)), tensor))

        c_array += '};'
        return c_array

    def tensor_to_const_array(self, name: str, tensor: torch.Tensor):
        return f'const {self.tensor_to_array(name, tensor)}'

    def dict_to_variables(self, variables_dict: dict):
        c_str = ''
        for key, value in variables_dict.items():
            data_type, data_value = value
            c_str += f'{data_type} {key} = {data_value};\n'
        return c_str

    def output_testing(self, output_len: int, expected_output: torch.Tensor):
        c_str = self.tensor_to_array('expectedOutput', expected_output)
        c_str += f"""
for (size_t i=0;i<{output_len};++i){{
    printf("Output [%d]: %f\\n", i, output[i]);
    assert(equalFloatDefault(output[i], expectedOutput[i]));
}}"""
        return c_str

    def conv_(self, output_len: int):
        return f"""float output[{output_len}];
CNN_ConvLayerForward_(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, input, weights, biases, output);"""

    def conv(self, output_len: int):
        return f"""float output[{output_len}];
CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);"""

    def conv_default(self, output_len: int):
        return f"""float output[{output_len}];
CNN_ConvLayerForwardDefault(inputChannels, inputHeight, inputWidth, outputChannels, kernel, input, weights, biases, output);"""

    def fc(self):
        return """float output[outputLen];
CNN_FcLayerForward(inputLen, outputLen, input, weights, biases, output);"""

    def max_pool_default(self, output_len: int):
        return f"""float output[{output_len}];
CNN_MaxPoolForwardDefault(inputChannels, inputHeight, inputWidth, kernel, input, output);"""

    def max_pool(self, output_len: int):
        return f"""float output[{output_len}];
CNN_MaxPoolForward(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, stride, padding, input, output);"""

    def max_pool_(self, output_len: int):
        return f"""float output[{output_len}];
CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, ceilMode, input, output);"""

    def prelu(self, output_len: int):
            return f"""float output[{output_len}];
CNN_PReLU(inputChannels, inputHeight, inputWidth, input, weights, output);"""

    def softmax(self, output_len: int):
        return f"""float output[{output_len}];
CNN_Softmax(inputLen, input, output);"""

    def softmax2d(self, output_len: int):
        return f"""float output[{output_len}];
CNN_Softmax2D(inputChannels, inputHeight, inputWidth, dim, input, output);"""

    def model(self, model: torch.nn.Module, input_size: tuple):
        model_c_str = ''
        i, j, k = 0, 0, 0
        for key, value in model.state_dict().items():
            name = key.split('.')[-1]
            if name == 'weight':
                name += str(j)
                j += 1
            elif name == 'bias':
                name += str(k)
                k += 1
            model_c_str += self.tensor_to_const_array(name, value) + '\n'
            i += 1

        i, j, k = 0, 0, 0

        def get_value(param: Union[int, tuple]):
            if isinstance(param, int):
                return param, param
            return param

        layer_str = ''
        output_channels = 3
        output_size = input_size
        output_len = output_channels * output_size[0] * output_size[1]
        for _, layer in enumerate(model.modules()):
            if i == 0:
                i += 1
                continue
            if i == 1:
                input_array_name = 'input'
            else:
                input_array_name = f'output{i-1}'
            name = layer.__class__.__name__
            input_size = output_size
            if name == 'Conv2d':
                kernel_h, kernel_w = get_value(layer.kernel_size)
                stride_h, stride_w = get_value(layer.stride)
                padding_h, padding_w = get_value(layer.padding)
                output_len, output_size = get_conv_output_size(layer.out_channels, input_size, layer.kernel_size, layer.stride, layer.padding)
                output_channels = layer.out_channels

                layer_str = f'float output{i}[{output_len}];\n' + \
                            f'CNN_ConvLayerForward_({layer.in_channels}, {input_size[0]}, {input_size[0]}, {output_channels}, {kernel_h}, {kernel_w}, {stride_h}, {stride_w}, {padding_h}, {padding_w}, {input_array_name}, weight{j}, bias{k}, output{i});\n'
                j += 1
                k += 1
            elif name == 'MaxPool2d':
                kernel_h, kernel_w = get_value(layer.kernel_size)
                stride_h, stride_w = get_value(layer.stride)
                padding_h, padding_w = get_value(layer.padding)
                output_len, output_size = get_max_pool_output_size(output_channels, input_size, layer.kernel_size, layer.stride, layer.padding, layer.ceil_mode)
                layer_str = f'float output{i}[{output_len}];\n' + \
                            f'CNN_MaxPoolForward_({output_channels}, {input_size[0]}, {input_size[1]}, {kernel_h}, {kernel_w}, {stride_h}, {stride_w}, {padding_h}, {padding_w}, {int(layer.ceil_mode)}, {input_array_name}, output{i});\n'
            elif name == 'Linear':
                output_len = layer.out_features
                output_channels = layer.out_features
                output_size = (1, 1)
                layer_str = f'float output{i}[{output_len}];\n' + \
                            f'CNN_FcLayerForward({layer.in_features}, {output_len}, {input_array_name}, weight{j}, bias{k}, output{i});\n'
                j += 1
                k += 1
            elif name == 'ReLU':
                layer_str = f'float output{i}[{output_len}];\n' + \
                            f'CNN_ReLU({output_len}, {input_array_name}, output{i});\n'
            elif name == 'PReLU':
                layer_str = f'float output{i}[{output_len}];\n' + \
                            f'CNN_PReLU({output_channels}, {input_size[0]}, {input_size[1]}, {input_array_name}, weight{j}, output{i});\n'
                j += 1
            elif name == 'Softmax':
                if input_size == (1, 1):
                    layer_str = f'float output{i}[{output_len}];\n' + \
                                f'CNN_Softmax({output_len}, {input_array_name}, output{i});\n'
                else:
                    layer_str = f'float output{i}[{output_len}];\n' + \
                                f'CNN_Softmax2D({output_channels}, {input_size[0]}, {input_size[1]}, {layer.dim-1}, {input_array_name}, output{i});\n'
            elif name == 'Flatten':
                layer_str = ''
                i -= 1
            i += 1
            model_c_str += layer_str
        return model_c_str
