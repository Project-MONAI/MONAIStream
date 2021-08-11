# DeepStream Cupy app for Monai UNet Model 

### Prerequisites:
- DeepStreamSDK 5.1
- Python 3.6
- Gst-python
- NumPy package
- OpenCV package
- CuPy for CUDA 11.1

Follow the steps in this README to run the python app [`deepstream_cupy_monai_unet.py`](deepstream_cupy_monai_unet.py).

### We will need to download:
- a mp4 file for endoscopy https://github.com/rijobro/real_time_seg/blob/main/example_data/EndoVis2017/d1_im.mp4
- a pytorch model https://github.com/rijobro/real_time_seg/blob/main/example_data/EndoVis2017/model_jit.pt


### Preparation steps: 
- Clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps into /opt/nvidia/deepstream/deepstream-5.1/sources/ on your machine
- Clone this repo monai-stream-experimental to be placed in /opt/nvidia/deepstream/deepstream-5.1/sources/deepstream_python_apps/apps/ on your machine
- Place the video somewhere you can mount to a container, current choice in app on host: /opt/nvidia/deepstream/deepstream-5.1/sources/deepstream_python_apps/apps/monai-stream-experimental/videos/d1_im.mp4
- Place the model somewhere you can mount to a container, current choice in app on host: /opt/nvidia/deepstream/deepstream-5.1/sources/deepstream_python_apps/apps/monai-stream-experimental/models/monai_unet_pytorch/1/model_jit.pt

### Start the DS Triton x86 container:
    
    $ xhost +
    $ docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /tmp/.X11-unix:/tmp/.X11-unix -v /opt/nvidia/deepstream/deepstream-5.1/sources/deepstream_python_apps/:/opt/nvidia/deepstream/deepstream/sources/python -v /opt/nvidia/deepstream/deepstream-5.1/samples/trtis_model_repo:/opt/nvidia/deepstream/deepstream/samples/trtis_model_repo -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.1  nvcr.io/nvidia/deepstream:5.1-21.02-triton


### Inside the container, install packages required by samples:
  
    $ apt update && apt install python3-gi python3-dev python3-gst-1.0 python3-opencv python3-numpy libgstrtspserver-1.0-0 gstreamer1.0-rtsp libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0 python3-matplotlib -y && pip3 install --upgrade opencv-python && pip3 install cupy-cuda111==8.6.0

### Set up the cupy library:
    $ cd /opt/nvidia/deepstream/deepstream-5.1/lib/ && mv pyds.so pyds.so.bkp && cp -r /opt/nvidia/deepstream/deepstream-5.1/sources/python/apps/monai-stream-experimental/lib/pyds.so ./ && python3 setup.py install

### Navigate to the app directory:
    $ cd /opt/nvidia/deepstream/deepstream-5.1/sources/python/apps/monai-stream-experimental


### Running the app:
    $ python3 sample/cupy-app/deepstream_cupy_monai_unet.py <uri>  <either debug or performance>
e.g.

    $ python3 sample/cupy-app/deepstream_cupy_monai_unet.py file:///opt/nvidia/deepstream/deepstream-5.1/sources/python/apps/monai-stream-experimental/videos/d1_im.mp4 debug
    $ python3 sample/cupy-app/deepstream_cupy_monai_unet.py file:///home/ubuntu/video1.mp4  debug
    $ python3 sample/cupy-app/deepstream_cupy_monai_unet.py rtsp://127.0.0.1/video1 debug
    $ python3 sample/cupy-app/deepstream_cupy_monai_unet.py rtsp://127.0.0.1/video1 performance
    $ python3 sample/cupy-app/deepstream_cupy_monai_unet.py file:///opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_720p.mp4 debug

The last arg in the command line decides whether the app will be printing info to the terminal, as well as saving mask info for debugging purposes to the folder output_frames/.

The Triton inference server config file is located at `configs/config_unet_pytorch_nopostprocess.txt` as specified in `deepstream_cupy_monai_unet.py`. This config file points to the location of the PyTorch model.


### Choice of config file:
The config file used by the app, which determines the model configuration, is specified in `deepstream_cupy_monai_unet.py` as a property of pgie. By default, it is set to `configs/config_unet_pytorch_nopostprocess.txt`.
