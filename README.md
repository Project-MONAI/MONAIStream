# MONAI Stream

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/Project-MONAI/monai-stream-experimental/actions/workflows/pr.yml/badge.svg)](https://github.com/Project-MONAI/monai-stream-experimental/actions/workflows/pr.yml)
[![Documentation Status](https://readthedocs.org/projects/monaistream/badge/?version=latest)](https://monaistream.readthedocs.io/en/latest/?badge=latest)


MONAI Stream SDK aims to equip experienced MONAI Researchers an Developers with the ability to
build streaming inference pipelines while enjoying the familiar MONAI development experience
and utilities.

## Features

> _The codebase is currently under active development._

- Framework to allow MONAI-style inference pipelines for streaming data.
- Compositional 

## Installation

### `x86` Development Container Setup

#### Creating a Local Development Container

To build a developer container for your workstation simply clone the repo and run the setup script as follows.

```bash
# clone the latest release from the repo
git clone -b main https://github.com/Project-MONAI/MONAIStream

# start development setup script
cd MONAIStream
./start_devel.sh
```

With the successful completion of the setup script, a container will be running containing all the necessary libraries
for the developer to start designing MONAI Stream SDK inference pipelines. The development however is limited to within
the container and the mounted volumes. The developer may modify ``Dockerfile.devel`` and ``start_devel.sh`` to suit their
needs.

#### Connecting VSCode to the Development Container

To start developing within the newly created MONAI Stream SDK development container users may choose to use their favorite
editor or IDE. Here, we show how one could setup VSCode on their local machine to start developing MONAI Stream inference
pipelines.

  1. Install [VSCode](https://code.visualstudio.com/download) on your Linux development workstation.
  2. Install the [Remote Development Extension pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) and restart VSCode.
  3. In VSCode select the icon ![VSCodeRDE](https://raw.githubusercontent.com/Project-MONAI/MONAIStream/main/docs/images/vscode_remote_development_ext.png) of the newly installed Remote Development extension on the left.
  4. Select "Containers" under "Remote Explorer" at the top of the dialog.
     ![VSCodeRemoteExplorer](https://raw.githubusercontent.com/Project-MONAI/MONAIStream/main/docs/images/vscode_remote_explorer.png)
  5. Attach to the MONAI Stream SDK container by clicking the "Attach to Container" icon ![VSCodeAttachContainer](https://raw.githubusercontent.com/Project-MONAI/MONAIStream/main/docs/images/vscode_attach_container.png) on the container name.

The above steps should allow the user to develop inside the MONAI Stream container using VSCode.

#### Run the Ultrasound Inference Sample App

MONAI Stream SDK comes with example inference pipelines. Here, we run a sample app
to perform instrument segmentation in an ultrasound video.

Inside the development container perform the following steps.

  1. Download the ultrasound data and models in the container.

    ```bash
    mkdir -p /app/data
    cd /app/data
    wget https://github.com/Project-MONAI/monai-stream-experimental/releases/download/data/US.zip
    unzip US.zip -d .
    ```

  2. Copy the ultrasound video to ``/app/videos/Q000_04_tu_segmented_ultrasound_256.avi`` as the example app expects.

    ```bash
    mkdir -p /app/videos
    cp /app/data/US/Q000_04_tu_segmented_ultrasound_256.avi /app/videos/.
    ```

  3. Convert ONNX model to TRT engine.

    ```bash
    cd /app/data/US/
    /usr/src/tensorrt/bin/trtexec --onnx=us_unet_256x256.onnx --saveEngine=model.engine --explicitBatch --verbose --workspace=5000
    ```

  4. Copy the ultrasound segmentation model under ``/app/models/us_unet_256x256/1`` as our sample app expects.

    ```bash
    mkdir -p /app/models/us_unet_256x256/1
    cp /app/data/US/model.engine /app/models/us_unet_256x256/1/.
    ```

  5. Running the example streaming bone scoliosis segmentation pipeline on the ultrasound video.
  
    ```bash
    cd /sample/monaistream-pytorch-pp-app
    python main.py
    ```

# Links

- Website: https://monai.io/
- API documentation: https://docs.monai.io/projects/stream
- Code: https://github.com/Project-MONAI/MONAIStream
- Project tracker: https://github.com/Project-MONAI/MONAIStream/projects
- Issue tracker: https://github.com/Project-MONAI/MONAIStream/issues
