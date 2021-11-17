# MONAI Stream

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/Project-MONAI/monai-stream-experimental/actions/workflows/pr.yml/badge.svg)](https://github.com/Project-MONAI/monai-stream-experimental/actions/workflows/pr.yml)
[![Documentation Status](https://readthedocs.org/projects/monaistream/badge/?version=latest)](https://monaistream.readthedocs.io/en/latest/?badge=latest)


MONAI Stream SDK aims to equip experienced MONAI Researchers an Developers with the ability to
build streaming inference pipelines while enjoying the familiar MONAI development experience
and utilities. 

MONAI Stream SDK natively supports:
- a number of input component types including real-time streams (RTSP), streaming URL, local video files,  
AJA Capture cards with direct memory access to GPU, and a Fake Source for testing purposes
- outputs components to allow the developer to view the result of their pipelines or just to test via Fake Sink,
- a number of filter types, including format conversion, video frame resizing and/or scaling, and most importantly a MONAI transform components
  that allows developers to plug-in MONAI transformations into the MONAI Stream pipeline.

![MONAIStreamArchitecture](https://raw.githubusercontent.com/Project-MONAI/MONAIStream/main/docs/images/MONAIStream_High-level_Architecture.svg)

MONAI Stream pipelines being with a source component, and end with a sink component, and the two are connect by a series of filter components.

[![](https://mermaid.ink/img/eyJjb2RlIjoiJSV7aW5pdDogeydzZWN1cml0eUxldmVsJzogJ2xvb3NlJywgJ3RoZW1lJzonYmFzZSd9fSUlXG5cbnN0YXRlRGlhZ3JhbS12MlxuICAgICAgc3RhdGUgVHJhbnNmb3JtQ2hhaW5Db21wb25lbnQge1xuICAgICAgICAgWypdIC0tPiBJbXBsaWNpdElucHV0TWFwcGluZ1xuICAgICAgICAgc3RhdGUgSW1wbGljaXRJbnB1dE1hcHBpbmcge1xuICAgICAgICAgICAgc3RhdGUgXCJbIElucHV0WzBdLCBJbnB1dFsxXSBdXCIgYXMgSU1JbnB1dHNcbiAgICAgICAgICAgIHN0YXRlIFwiezxicj4nb3JpZ2luYWxfaW1hZ2UnOiBJbnB1dFswXSw8YnI-ICdzZWdfb3V0cHV0JzogSW5wdXRbMV08YnI-fVwiIGFzIElNT3V0cHV0c1xuICAgICAgICAgICAgWypdIC0tPiBJTUlucHV0c1xuICAgICAgICAgICAgSU1JbnB1dHMgLS0-IElNT3V0cHV0c1xuICAgICAgICAgICAgSU1PdXRwdXRzIC0tPiBbKl1cbiAgICAgICAgIH1cbiAgICAgICAgIEltcGxpY2l0SW5wdXRNYXBwaW5nIC0tPiBBY3RpdmF0aW9uc2RcbiAgICAgICAgIEFjdGl2YXRpb25zZCAtLT4gQXNEaXNjcmV0ZWRcbiAgICAgICAgIEFzRGlzY3JldGVkIC0tPiBBZGRDaGFubmVsZFxuICAgICAgICAgQWRkQ2hhbm5lbGQgLS0-IEFzQ2hhbm5lbExhc3RkXG4gICAgICAgICBBc0NoYW5uZWxMYXN0ZCAtLT4gQ29uY2F0SXRlbXNkXG4gICAgICAgICBDb25jYXRJdGVtc2QgLS0-IExhbWJkYWRcbiAgICAgICAgIExhbWJkYWQgLS0-IENhc3RUb1R5cGVkXG4gICAgICAgICBDYXN0VG9UeXBlZCAtLT4gSW1wbGljaXRPdXRwdXRNYXBwaW5nXG4gICAgICAgICBzdGF0ZSBJbXBsaWNpdE91dHB1dE1hcHBpbmcge1xuICAgICAgICAgICAgICAgc3RhdGUgXCJ7PGJyPidvcmlnaW5hbF9pbWFnZSc6IE91dHB1dFswXSw8YnI-ICdzZWdfb3V0cHV0JzogT3V0cHV0WzFdLDxicj4nc2VnX292ZXJsYXknOiBPdXRwdXRbMl08YnIvPn1cIiBhcyBPTUlucHV0c1xuICAgICAgICAgICAgICAgc3RhdGUgXCJPdXRwdXRbM11cIiBhcyBPTU91dHB1dFxuICAgICAgICAgICAgICAgWypdIC0tPiBPTUlucHV0c1xuICAgICAgICAgICAgICAgT01JbnB1dHMgLS0-IE9NT3V0cHV0c1xuICAgICAgICAgICAgICAgT01PdXRwdXRzIC0tPiBbKl1cbiAgICAgICAgIH1cbiAgICAgICAgIEltcGxpY2l0T3V0cHV0TWFwcGluZyAtLT4gWypdXG4gICAgICB9IiwibWVybWFpZCI6eyJ0aGVtZSI6ImRhcmsifSwidXBkYXRlRWRpdG9yIjp0cnVlLCJhdXRvU3luYyI6dHJ1ZSwidXBkYXRlRGlhZ3JhbSI6dHJ1ZX0)](https://mermaid.live/edit#eyJjb2RlIjoiJSV7aW5pdDogeydzZWN1cml0eUxldmVsJzogJ2xvb3NlJywgJ3RoZW1lJzonYmFzZSd9fSUlXG5cbnN0YXRlRGlhZ3JhbS12MlxuICAgICAgc3RhdGUgVHJhbnNmb3JtQ2hhaW5Db21wb25lbnQge1xuICAgICAgICAgWypdIC0tPiBJbXBsaWNpdElucHV0TWFwcGluZ1xuICAgICAgICAgc3RhdGUgSW1wbGljaXRJbnB1dE1hcHBpbmcge1xuICAgICAgICAgICAgc3RhdGUgXCJbIElucHV0WzBdLCBJbnB1dFsxXSBdXCIgYXMgSU1JbnB1dHNcbiAgICAgICAgICAgIHN0YXRlIFwiezxicj4nb3JpZ2luYWxfaW1hZ2UnOiBJbnB1dFswXSw8YnI-ICdzZWdfb3V0cHV0JzogSW5wdXRbMV08YnI-fVwiIGFzIElNT3V0cHV0c1xuICAgICAgICAgICAgWypdIC0tPiBJTUlucHV0c1xuICAgICAgICAgICAgSU1JbnB1dHMgLS0-IElNT3V0cHV0c1xuICAgICAgICAgICAgSU1PdXRwdXRzIC0tPiBbKl1cbiAgICAgICAgIH1cbiAgICAgICAgIEltcGxpY2l0SW5wdXRNYXBwaW5nIC0tPiBBY3RpdmF0aW9uc2RcbiAgICAgICAgIEFjdGl2YXRpb25zZCAtLT4gQXNEaXNjcmV0ZWRcbiAgICAgICAgIEFzRGlzY3JldGVkIC0tPiBBZGRDaGFubmVsZFxuICAgICAgICAgQWRkQ2hhbm5lbGQgLS0-IEFzQ2hhbm5lbExhc3RkXG4gICAgICAgICBBc0NoYW5uZWxMYXN0ZCAtLT4gQ29uY2F0SXRlbXNkXG4gICAgICAgICBDb25jYXRJdGVtc2QgLS0-IExhbWJkYWRcbiAgICAgICAgIExhbWJkYWQgLS0-IENhc3RUb1R5cGVkXG4gICAgICAgICBDYXN0VG9UeXBlZCAtLT4gSW1wbGljaXRPdXRwdXRNYXBwaW5nXG4gICAgICAgICBzdGF0ZSBJbXBsaWNpdE91dHB1dE1hcHBpbmcge1xuICAgICAgICAgICAgICAgc3RhdGUgXCJ7PGJyPidvcmlnaW5hbF9pbWFnZSc6IE91dHB1dFswXSw8YnI-ICdzZWdfb3V0cHV0JzogT3V0cHV0WzFdLDxicj4nc2VnX292ZXJsYXknOiBPdXRwdXRbMl08YnIvPn1cIiBhcyBPTUlucHV0c1xuICAgICAgICAgICAgICAgc3RhdGUgXCJPdXRwdXRbM11cIiBhcyBPTU91dHB1dFxuICAgICAgICAgICAgICAgWypdIC0tPiBPTUlucHV0c1xuICAgICAgICAgICAgICAgT01JbnB1dHMgLS0-IE9NT3V0cHV0c1xuICAgICAgICAgICAgICAgT01PdXRwdXRzIC0tPiBbKl1cbiAgICAgICAgIH1cbiAgICAgICAgIEltcGxpY2l0T3V0cHV0TWFwcGluZyAtLT4gWypdXG4gICAgICB9IiwibWVybWFpZCI6IntcbiAgXCJ0aGVtZVwiOiBcImRhcmtcIlxufSIsInVwZGF0ZUVkaXRvciI6dHJ1ZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOnRydWV9)

Currently, MONAI Stream SDK provides a compatibility layer to [Deepstream SDK](https://developer.nvidia.com/deepstream-sdk) so
MONAI developers may be able to plug-in existing MONAI inference code into a MONAI Stream pipeline.

## Features

> _The codebase is currently under active development._

- Framework to allow MONAI-style inference pipelines for streaming data.
- Allows for MONAI chained transformations to be used on streaming data.
- Inference models can be used natively in MONAI or deployed via [Triton Inference Server](https://github.com/triton-inference-server/server).
- Natively provides support for _x86_ and [Clara AGX](https://developer.nvidia.com/clara-holoscan-sdk) architectures
  - with the future aim to allow developers to deploy the same code in both architectures with no changes.

## Getting Started: `x86` Development Container Setup

### Creating a Local Development Container

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

### Connecting VSCode to the Development Container

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

### Run the Ultrasound Inference Sample App

MONAI Stream SDK comes with example inference pipelines. Here, we run a sample app
to perform instrument segmentation in an ultrasound video.

Inside the development container perform the following steps.

  1. Download the ultrasound data and models in the container.

    mkdir -p /app/data
    cd /app/data
    wget https://github.com/Project-MONAI/monai-stream-experimental/releases/download/data/US.zip
    unzip US.zip -d .

  2. Copy the ultrasound video to ``/app/videos/Q000_04_tu_segmented_ultrasound_256.avi`` as the example app expects.

    mkdir -p /app/videos
    cp /app/data/US/Q000_04_tu_segmented_ultrasound_256.avi /app/videos/.

  3. Convert ONNX model to TRT engine.

    cd /app/data/US/
    /usr/src/tensorrt/bin/trtexec --onnx=us_unet_256x256.onnx --saveEngine=model.engine --explicitBatch --verbose --workspace=5000

  4. Copy the ultrasound segmentation model under ``/app/models/us_unet_256x256/1`` as our sample app expects.

    mkdir -p /app/models/us_unet_256x256/1
    cp /app/data/US/model.engine /app/models/us_unet_256x256/1/.

  5. Running the example streaming bone scoliosis segmentation pipeline on the ultrasound video.
  
    cd /sample/monaistream-pytorch-pp-app
    python main.py

# Links

- Website: https://monai.io/
- API documentation: https://docs.monai.io/projects/stream
- Code: https://github.com/Project-MONAI/MONAIStream
- Project tracker: https://github.com/Project-MONAI/MONAIStream/projects
- Issue tracker: https://github.com/Project-MONAI/MONAIStream/issues
