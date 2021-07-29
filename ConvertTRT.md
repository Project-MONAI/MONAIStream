### Export the pytorch model into ONNX
Run  `python src/utils/convert_pytorch2onnx.py` to generate the ONNX model. Modify the path to the PyTorch model before running the script. The PyTorch model can be downloaded from https://github.com/rijobro/real_time_seg/blob/main/example_data/EndoVis2017/model_jit.pt.

### Get the tensorrt container
    docker pull nvcr.io/nvidia/tensorrt:20.11-py3
### Run the container
    docker run -it --rm -v /[directory-containing-the-pytorch-model]:/workspace/convert2trt nvcr.io/nvidia/tensorrt:20.11-py3
### Install polygraphy within the container 
We need to do this because the onnx model has some loops that will throw an error when exporting to trt.

    pip install --upgrade pip
    pip install nvidia-pyindex 
    pip install --upgrade onnx 
    pip install onnx-graphsurgeon  
    pip install --upgrade polygraphy
    pip install colored 
    pip install onnxruntime
### Fix loops in onnx model for trt export 
    polygraphy surgeon sanitize monai_unet_pyt.onnx --fold-constants --output model_unet_pyt_folded.onnx
### Finally export the ONNX model into a TRT engine
    trtexec --onnx=model_unet_pyt_folded.onnx --saveEngine=monai_unet.engine --explicitBatch --verbose --workspace=1000

After we have the tensorrt engine, we will place it in [repo]/models/monai_unet_trt/1/.