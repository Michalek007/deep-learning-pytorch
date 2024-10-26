import torch
from torch import nn
import torch.nn.functional as F


class CParser:
    def __init__(self):
        self.flatten = nn.Flatten(start_dim=0)

    def to_array(self, name: str, tensor: torch.Tensor):
        tensor = self.flatten(tensor)
        if tensor.dtype == torch.float:
            data_type = 'float'
        elif tensor.dtype == torch.int:
            data_type = 'int'
        else:
            data_type = 'undefined'
        c_array = f'{data_type} {name}[] = {{'

        c_array += ', '.join(map(lambda t: str(t.item()), tensor))
        # for value in tensor:
        #     c_array += str(value.item()) + ', '

        c_array += '};'
        return c_array

    def from_dict_to_variables(self, variables_dict: dict):
        c_str = ''
        for key, value in variables_dict.items():
            data_type, data_value = value
            c_str += f'{data_type} {key} = {data_value};\n'
        return c_str

    def conv_testing(self, output_size: int):
        return f'float output[{output_size}];' + \
               '\nCNN_ConvLayerForward(inputChannels, inputWidth, inputHeight, outputChannels, kernelWidth, kernelHeight, stride, padding, input, weights, biases, output);' + \
               f'\nfor (size_t i=0;i<{output_size};++i){{' + \
               '\n    printf("Output: %f\\n", output[i]);' + \
               '\n}'

    def fc_testing(self):
        return """float output[outputLen];
CNN_FcLayerForward(inputLen, outputLen, input, weights, biases, output);

for (size_t i=0;i<outputLen;++i){
    printf("Output: %f\\n", output[i]);
}"""

    def max_pool_testing(self, output_size: int):
        return f"""float output[{output_size}];
CNN_MaxPoolForwardDefault(inputChannels, inputWidth, inputHeight, kernel, input, output);
for (size_t i=0;i<{output_size};++i){{
    printf("Output: %f\\n", output[i]);
}}"""

    def from_model(self, model):
        model_c_str = ''
        i, j = 0, 0
        for key, value in model.state_dict().items():
            name = key.split('.')[-1] + str(j)
            model_c_str += self.to_array(name, value) + '\n'
            if i % 2 != 0:
                j += 1
            i += 1

        i, j = 0, 0
        last_output = 0
        for name, values in model.layers():
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
