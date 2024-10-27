import torch
from torch import nn
import torch.nn.functional as F


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

    def conv(self, output_len: int):
        return f"""float output[{output_len}];
CNN_ConvLayerForward(inputChannels, inputWidth, inputHeight, outputChannels, kernelWidth, kernelHeight, stride, padding, input, weights, biases, output);"""

    def fc(self):
        return """float output[outputLen];
CNN_FcLayerForward(inputLen, outputLen, input, weights, biases, output);"""

    def max_pool_default(self, output_len: int):
        return f"""float output[{output_len}];
CNN_MaxPoolForwardDefault(inputChannels, inputWidth, inputHeight, kernel, input, output);"""

    def model(self, model: torch.nn.Module):
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
        for name, values in model.layers:
            # print(name)
            # print(values)
            if i == 0:
                output_array_name = 'input'
            else:
                output_array_name = f'output{i}'

            if name == 'conv':
                layer_str = f'float output{i+1}[{values["output_len"]}];\n' +\
                            f'CNN_ConvLayerForwardDefault({values["in_channels"]}, {values["input_width"]}, {values["input_height"]}, {values["out_channels"]}, {values["kernel"]}, {output_array_name}, weight{j}, bias{j}, output{i+1});\n'
                j += 1
            elif name == 'max_pool':
                layer_str = f'float output{i+1}[{values["output_len"]}];\n' +\
                            f'CNN_MaxPoolForwardDefault({values["in_channels"]}, {values["input_width"]}, {values["input_height"]}, {values["kernel"]}, output{i}, output{i+1});\n'
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
