################################################################################
# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GstAudio", "1.0")

from gi.repository import Gst

Gst.init(None)

from . import _version

version_dict = _version.get_versions()

__version__ = version_dict.get("version", "0+unknown")
__revision_id__ = version_dict.get("full-revisionid")
__copyright__ = "(c) 2021 MONAI Consortium"

del version_dict


def print_config(file=sys.stdout):

    from collections import OrderedDict

    import numpy as np
    import torch

    output = OrderedDict()
    output["MONAIStream"] = __version__
    output["Numpy"] = np.version.full_version
    output["Pytorch"] = torch.__version__

    print(__copyright__)

    for k, v in output.items():
        print(f"{k} version: {v}", file=file, flush=True)

    print(f"MONAIStream rev id: {__revision_id__}")
