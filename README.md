# Deepstream for Monai

This repo will contain the work needed to do POC to provide framework where Monai pipelines can be run with Deepstream backend.

There are the following Python DeepStream sample applications:

 - [`deepstream_cupy_monai_unet`](sample//cupy-app/deepstream_cupy_monai_unet.py) which takes in a mp4 file and performs inference using the Triton Inference Server backend in DeepStream, then displays the segmentation mask on screen.


For converting the PyTorch model into TRT, follow the steps in [`ConvertTRT.md`](ConvertTRT.md).