============
Installation
============

MONAI Stream SDK is a wrapper for `DeepStream SDK <https://developer.nvidia.com/deepstream-sdk>`_,
and as such it requires DeepStream to be able to run. Users may choose to install all the libraries
required by DeepStream on their machine as well as DeepStream SDK by following
`this <https://developer.nvidia.com/deepstream-getting-started>`_ guide, however, MONAI Stream SDK
provides a dockerfile script that will automatically perform the setup and allow the user to developer
inside a container on the machine of their choice (x86 or `Clara AGX <https://developer.nvidia.com/clara-agx-devkit>`_)
running a Linux operating system.

Steps for `x86` Development Container Setup
===========================================

Creating a Local Development Container
--------------------------------------

To build a developer container for your workstation simply clone the repo and run the setup script as follows.

.. code-block:: bash

    # clone the latest release from the repo
    git clone -b <release_tag> https://github.com/Project-MONAI/MONAIStream

    # start development setup script
    cd MONAIStream
    ./start_devel.sh

With the successful completion of the setup script, a container will be running containing all the necessary libraries
for the developer to start designing MONAI Stream SDK inference pipelines. The development however is limited to within
the container and the mounted volumes. The developer may modify ``Dockerfile.devel`` and ``start_devel.sh`` to suit their
needs.

Connecting VSCode to the Development Container
----------------------------------------------

To start developing within the newly created MONAI Stream SDK development container users may choose to use their favorite
editor or IDE. Here, we show how one could setup VSCode on their local machine to start developing MONAI Stream inference
pipelines.

  1. Install `VSCode <https://code.visualstudio.com/download>`_ on your Linux development workstation.
  2. Install the `Remote Development Extension pack <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack>`_ and restart VSCode.
  3. In VSCode select the icon |VSCodeRDE| of the newly installed Remote Development extension on the left.
  4. Select "Containers" under "Remote Explorer" at the top of the dialog.
     |VSCodeRemoteExplorer|
  5. Attach to the MONAI Stream SDK container by clicking the "Attach to Container" icon |VSCodeAttachContainer| on the container name.

  .. |VSCodeRDE| image:: ../images/vscode_remote_development_ext.png
    :alt: VSCode Remote Development Extension Icon

  .. |VSCodeRemoteExplorer| image:: ../images/vscode_remote_explorer.png
    :alt: VSCode Remote Development Extension Icon

  .. |VSCodeAttachContainer| image:: ../images/vscode_attach_container.png
    :alt: VSCode Remote Development Extension Icon

The above steps should allow the user to develop inside the MONAI Stream container using VSCode.

Run the Ultrasound Inference Sample App
---------------------------------------

MONAI Stream SDK comes with example inference pipelines. Here, we run a sample app
to perform bone scoliosis segmentation in an ultrasound video.

Inside the development container perform the following steps.

  1. Download the ultrasound data and models in the container.

  .. code-block:: bash
  
    mkdir -p /app/data
    cd /app/data
    wget https://github.com/Project-MONAI/MONAIStream/releases/download/data/US.zip
    unzip US.zip -d .

  2. Copy the ultrasound video to ``/app/videos/Q000_04_tu_segmented_ultrasound_256.avi`` as the example app expects.

    .. code-block:: bash
    
      mkdir -p /app/videos
      cp /app/data/US/Q000_04_tu_segmented_ultrasound_256.avi /app/videos/.

  3. Convert PyTorch or ONNX model to TRT engine.

      a. To Convert the provided ONNX model to a TRT engine use:

      .. code-block:: bash

          cd /app/data/US/
          /usr/src/tensorrt/bin/trtexec --onnx=us_unet_256x256.onnx --saveEngine=model.engine --explicitBatch --verbose --workspace=5000
      
      b. To convert the PyTorch model to a TRT engine use:

      .. code-block:: bash

          cd /app/data/US/
          monaistream convert -i us_unet_jit.pt -o monai_unet.engine -I INPUT__0 -O OUTPUT__0 -S 1 3 256 256

  4. Copy the ultrasound segmentation model under ``/app/models/monai_unet_trt/1`` as our sample app expects.

    .. code-block:: bash
    
      mkdir -p /app/models/monai_unet_trt/1
      cp /app/data/US/monai_unet.engine /app/models/monai_unet_trt/1/.
      cp /app/data/US/config_us_trt.pbtxt /app/models/monai_unet_trt/config.pbtxt

  5. Now we are ready to run the example streaming ultrasound bone scoliosis segmentation pipeline.
  
    .. code-block:: bash
    
        cd /sample/monaistream-pytorch-pp-app
        python main.py

      .. code-block:: bash

          cd /sample/monaistream-pytorch-pp-app
          python main.py


Steps for `Clara AGX Developer Kit` Development Setup
=====================================================

Setting Up Clara AGX Developer Kit
----------------------------------

To setup the Clara AGX developer kit, use `Clara Holoscan SDK v0.1 <https://developer.nvidia.com/clara-holoscan-sdk>`_ to install the required components. MONAI Stream is only supported on Clara AGX Developer Kit in dGPU configuration.

The SDK Manager will flash the system for iGPU configuration, to get dGPU configuration and related installations, please follow chapter `Switching Between iGPU and dGPU <https://docs.nvidia.com/clara-holoscan/sdk-user-guide/dgpu_setup.html>`_ in latest Clara Holoscan SDK docs.

Once dGPU mode is enabled, set up the m2 SSD as described in `Storage Setup <https://docs.nvidia.com/clara-holoscan/sdk-user-guide/storage_setup.html>`_ to ensure that the AGX disk is correctly partitioned and mounted. 

Now, prepare DeepStream to use Triton:

  1. Install required packages.

      .. code-block:: bash

        sudo apt update && sudo apt-get install ffmpeg libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-alsa libgstreamer1.0-dev libgstrtspserver-1.0-dev libx11-dev libjson-glib-dev

  2. Run :code:`prepare_ds_trtis_model_repo.sh`.

      .. code-block:: bash

        cd /opt/nvidia/deepstream/deepstream-6.0/samples
        sudo ./prepare_ds_trtis_model_repo.sh

      .. NOTE:: :code:`prepare_ds_trtis_model_repo.sh` can take few minutes to complete.

  3. Currently, TensorFlow is not supported on Clara AGX Developer Kit in dGPU configuration. So, move the folders to avoid errors related to TensorFlow.

      .. code-block:: bash

        cd /opt/nvidia/deepstream/deepstream-6.0/lib/triton_backends
        sudo mv tensorflow1/ tensorflow1_bkup/
        sudo mv tensorflow2/ tensorflow2_bkup/

Next, setup the environement to use MONAI Stream:

  1. Install required apt packages.

      .. code-block:: bash

        sudo apt update
        sudo apt install -y python3-pip python3-gi python3-dev python3-gst-1.0 python3-opencv python3-venv python3-numpy libgstrtspserver-1.0-0 libgstreamer-plugins-base1.0-dev gstreamer1.0-rtsp gstreamer1.0-tools gstreamer1.0-libav libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0 gstreamer1.0-plugins-base gstreamer1.0-python3-plugin-loader

  2. Install Python packages using pip.

      .. code-block:: bash

        pip3 install --upgrade pip
        pip3 install --upgrade opencv-python
        pip3 install Cython
        pip3 install numpy==1.19.4
        pip3 install cupy
        pip3 install torchvision jinja2 pydantic monai

      .. NOTE:: Installing :code:`cupy` can take few minutes.

  3. Clone MONAI Stream repo

      .. code-block:: bash

        git clone git@github.com:Project-MONAI/MONAIStream.git /app
        cd /app

  4. Set up DeepStream Python bindings.

      .. code-block:: bash

        sudo cp /app/lib/pyds-py3.6-cagx.so /opt/nvidia/deepstream/deepstream-6.0/lib/pyds.so
        sudo chown -R $USER /usr/local/lib/python3.6/dist-packages/
        cd /opt/nvidia/deepstream/deepstream-6.0/lib
        sudo python3 setup.py install
        cd -

.. NOTE:: The steps to run the Ultrasound inference sample app is same as on x86 machine. Please follow `Run the Ultrasound Inference Sample App` in `Steps for `x86` Development Container Setup` section.

Setting Up AJA Capture Card
---------------------------

Setting up AJA capture cards is an optional step for MONAI Stream. To setup AJA capture card on Clara AGX Developer Kit, follow chapter `AJA Video System <https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html>`_ in latest Clara Holoscan SDK docs.

Running the AJA Capture Sample App
----------------------------------

To run a sample app to do RDMA capture using AJA capture card, use the following steps.

  1. Verify :code:`ajavideosrc` gst-plugin is setup properly.

      .. code-block:: bash

        gst-inspect-1.0 ajavideosrc

  2. If step 1 outputs the details about :code:`ajavideosrc` gst-plugin, then run the sample app. This step will output live video on display.

      .. code-block:: bash

        PYTHONPATH=src/ python3 sample/monaistream-rdma-capture-app/main.py
