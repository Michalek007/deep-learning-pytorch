

def get_conv_output_size(out_channels: int, input_width: int, input_height: int, kernel_size: int, stride: int = 1, padding: int = 0):
    output_width = (input_width - kernel_size + 2 * padding)/stride + 1
    output_height = (input_height - kernel_size + 2 * padding)/stride + 1
    return int(out_channels * output_width * output_height), int(output_width), int(output_height)


def get_max_pool_output_size(in_channels: int, input_width: int, input_height: int, kernel_size: int, stride: int = None):
    if not stride:
        stride = kernel_size
    output_width = (input_width - kernel_size)/stride + 1
    output_height = (input_height - kernel_size)/stride + 1
    return int(in_channels * output_width * output_height), int(output_width), int(output_height)
