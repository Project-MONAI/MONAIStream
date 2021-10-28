=================
Development Setup
=================

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
    git clone -b main https://github.com/Project-MONAI/MONAIStream

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

Run the Endoscopy Inference Sample App
--------------------------------------

MONAI Stream SDK comes with example inference pipelines. Here, we run a sample app
to perform instrument segmentation in an endoscopy video.

Inside the development container perform the following steps.

  1. Download the endoscopy data and models in the container.

  .. code-block:: bash
  
    mkdir -p /app/data
    cd /app/data
    wget https://github.com/Project-MONAI/monai-stream-experimental/releases/download/data/CholecSeg8K.zip
    unzip CholecSeg8K.zip -d .

  2. Copy the endoscopy video to ``/app/videos/endo.mp4`` as the example app expects.

  .. code-block:: bash

    mkdir -p /app/videos
    cp /app/data/CholecSeg8K/endo.mp4 /app/videos/.

  3. Convert ONNX model to TRT engine.

  .. code-block:: bash

      cd /app/data/CholecSeg8K/
      /usr/src/tensorrt/bin/trtexec --onnx=cholec_unet_864x480.onnx --saveEngine=model.engine --explicitBatch --verbose --workspace=1000

  4. Copy the endoscopy instrument segmentation model under ``/app/models/cholec_unet_864x480/1`` as our sample app expects.

  .. code-block:: bash
  
    mkdir -p /app/models/cholec_unet_864x480/1
    cp /app/data/CholecSeg8K/model.engine /app/models/cholec_unet_864x480/1/.

  5. Running the example streaming instrument segmentation pipeline on the endoscopy video.
  
  .. code-block:: bash
  
      cd /sample/monaistream-pytorch-pp-app
      python main.py


Steps for `Clara AGX` Development Setup
=======================================

Setting Up Clara AGX
--------------------

<docs here>

Setting Up AJA Capture
----------------------

<docs here>

Running the AJA Capture Sample App
----------------------------------

<docs here>
