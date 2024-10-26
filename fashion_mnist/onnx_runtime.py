import onnxruntime
import torch
from core import NeuralNetwork


model = NeuralNetwork()
model.load_state_dict(torch.load('fashion_mnist.pth', weights_only=True))
model.eval()

torch_input = torch.randn(1, 1, 28, 28)
onnx_program = torch.onnx.dynamo_export(model, torch_input)

onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
print(f"Input length: {len(onnx_input)}")
print(f"Sample input: {onnx_input}")

ort_session = onnxruntime.InferenceSession("./fashion_mnist.onnx", providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

torch_outputs = model(torch_input)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"ONNX output: {onnxruntime_outputs}")
print(f"Torch output: {torch_outputs}")
