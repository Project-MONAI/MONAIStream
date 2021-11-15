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

### `x86` Development Container Setup

#### Creating a Local Development Container


#### Connecting VSCode to the Development Container


#### Run the Ultrasound Inference Sample App


# Links

- Website: https://monai.io/
- API documentation: https://docs.monai.io/projects/stream
- Code: https://github.com/Project-MONAI/MONAIStream
- Project tracker: https://github.com/Project-MONAI/MONAIStream/projects
- Issue tracker: https://github.com/Project-MONAI/MONAIStream/issues
