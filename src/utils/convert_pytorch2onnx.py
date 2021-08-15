from torch import nn
import torch.onnx

model_path = "model_jit.pt"  # modify with your own path to the pytorch model


torch_model = torch.jit.load(model_path)
torch_model = torch_model.eval()

x = torch.randn(1, 3, 1024, 1264)
torch_out = torch_model(x)
input_names = ["INPUT__0"]
output_names = ["OUTPUT__0"]
# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "monai_unet_pyt.onnx",   # where to save the model (can be a file or file-like object)
                  example_outputs=torch_out,
                  # export_params=True,        # store the trained parameter weights inside the model file
                  verbose=True,
                  input_names=input_names, output_names=output_names
                  )