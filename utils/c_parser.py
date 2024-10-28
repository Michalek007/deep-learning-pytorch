import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple

from nn_utils import get_conv_output_size, get_max_pool_output_size


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

        c_array += ', '.join(map(lambda t: str(t.item()), tensor))

        c_array += '};'
        return c_array

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
CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, input, output);"""

    def model(self, model: torch.nn.Module, from_layers_property: bool = True):
        model_c_str = ''
        i, j = 0, 0
        for key, value in model.state_dict().items():
            name = key.split('.')[-1] + str(j)
            model_c_str += self.tensor_to_array(name, value) + '\n'
            if i % 2 != 0:
                j += 1
            i += 1

        i, j = 0, 0
        last_output = 0

        if not from_layers_property:
            def get_value(param: Union[int, tuple]):
                if isinstance(param, int):
                    return param
                if param[0] == param[1]:
                    return param[0]
                return None

            for i, layer in enumerate(model.modules()):
                if i == 0:
                    continue
                name = layer.__class__.__name__
                print()
                print(name)
                if name == 'Conv2d':
                    kernel_size = get_value(layer.kernel_size)
                    stride = get_value(layer.stride)
                    padding = get_value(layer.padding)

                    print(layer.in_channels)
                    print(layer.out_channels)
                elif name == 'MaxPool2d':
                    kernel_size = get_value(layer.kernel_size)
                    stride = get_value(layer.stride)
                    padding = get_value(layer.padding)

                elif name == 'ReLU':
                    pass
                elif name == 'Linear':
                    print(layer.in_features)
                    print(layer.out_features)

        for name, values in model.layers:
            # print(name)
            # print(values)
            if i == 0:
                output_array_name = 'input'
            else:
                output_array_name = f'output{i}'

            if name == 'conv':
                layer_str = f'float output{i+1}[{values["output_len"]}];\n' +\
                            f'CNN_ConvLayerForwardDefault({values["in_channels"]}, {values["input_height"]}, {values["input_width"]}, {values["out_channels"]}, {values["kernel"]}, {output_array_name}, weight{j}, bias{j}, output{i+1});\n'
                j += 1
            elif name == 'max_pool':
                layer_str = f'float output{i+1}[{values["output_len"]}];\n' +\
                            f'CNN_MaxPoolForwardDefault({values["in_channels"]}, {values["input_height"]}, {values["input_width"]}, {values["kernel"]}, output{i}, output{i+1});\n'
            elif name == 'fc':
                layer_str = f'float output{i+1}[{values["output_len"]}];\n' +\
                            f'CNN_FcLayerForward({values["input_len"]}, {values["output_len"]}, output{i}, weight{j}, bias{j}, output{i+1});\n'
                j += 1
            elif name == 'relu':
                layer_str = f'float output{i+1}[{last_output}];\n' +\
                            f'CNN_ReLU({last_output}, output{i}, output{i+1});\n'
            else:
                layer_str = ''

            try:
                last_output = values["output_len"]
            except KeyError:
                pass
            model_c_str += layer_str
            i += 1
        return model_c_str
