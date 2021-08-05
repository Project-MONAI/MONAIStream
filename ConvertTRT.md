## Step 1 Export the pytorch model into ONNX (Optional)
Run  `python src/utils/convert_pytorch2onnx.py` to generate the ONNX model. Modify the path to the PyTorch model before running the script. The PyTorch model can be downloaded from https://github.com/rijobro/real_time_seg/blob/main/example_data/EndoVis2017/model_jit.pt.

## Step 2 Set up container environment
### Pull the tensorrt container
    docker pull nvcr.io/nvidia/tensorrt:20.11-py3
### Run the container
    docker run -it --rm -v /[directory-containing-the-pytorch-or-onnx-model]:/somewhere nvcr.io/nvidia/tensorrt:20.11-py3

## Step 3 Use polygraphy to clean up ONNX model before exporting to TensorRT (Optional)
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

## Step 4 Export the ONNX model to TensorRT 

If you've skipped step 1 and 3, you can download the folded ONNX model at [`models/monai_unet_onnx/model_unet_pyt_folded.onnx`](models/monai_unet_onnx/model_unet_pyt_folded.onnx), since ONNX models are platform agnostic while TensorRT engines are platform specific.

### Finally export the ONNX model into a TRT engine
This step is taken also in the container run in step 2.

    trtexec --onnx=model_unet_pyt_folded.onnx --saveEngine=monai_unet.engine --explicitBatch --verbose --workspace=1000

After we have the tensorrt engine, we will place it in [path-to-this-repo]/models/monai_unet_trt/1/. 
